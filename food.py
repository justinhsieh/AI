import streamlit as st
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import google.generativeai as genai
import time
import os

# 請確認這裡填入的是正確的 Key
API_KEY = "AIzaSyAGrZW9H2_02Dmy7i3NEj9eTWyN1jO9zvo" 

# 立即配置 Gemini，確保後續函式都能抓到 Key
if API_KEY and "填在這裡" not in API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"API Key 設定失敗: {e}")

try:
    cuda.init()
except:
    pass

LABELS = [
    'Beef', 'Bell Pepper', 'Broccoli', 'Cabbage', 'Carrot',
    'Chicken', 'Cucumber', 'Eggplant', 'Mushroom', 'Onion',
    'Pork', 'Potato', 'Tomato'
]

@st.cache_resource
def load_trt_model(engine_path="best.engine"):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    try:
        dev = cuda.Device(0)
    except Exception as e:
        st.error("無法存取 GPU，請確認 CUDA 是否正常。")
        raise e

    cfx = dev.make_context()
    
    if not os.path.exists(engine_path):
        cfx.pop()
        raise FileNotFoundError(f"找不到模型檔案: {engine_path}，請確認檔案位置。")

    try:
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
    except Exception as e:
        cfx.pop()
        raise e

    cfx.pop()
    return cfx, engine, context

# 在主程式中載入資源
try:
    cfx, engine, context = load_trt_model()
except Exception as e:
    st.error(f"模型載入失敗: {e}")
    st.stop()

def preprocess(img, input_size=(640, 640)):
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = input_size

    # 計算縮放比例 (保持長寬比)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 計算縮放後的尺寸
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    
    # 計算需要補多少黑邊 (Padding)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # divide padding into 2 sides

    # 1. Resize
    if shape[::-1] != new_unpad:
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    # 2. Add Border (補黑邊)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 3. 轉成 TensorRT 需要的格式
    img_rgb = img_padded[:, :, ::-1]  # BGR to RGB
    img_transposed = img_rgb.transpose(2, 0, 1)  # HWC to CHW
    img_out = np.ascontiguousarray(img_transposed).astype(np.float32) / 255.0
    
    return img_out[np.newaxis, :, :, :], (r, r), (dw, dh)

def infer(img):
    input_image, ratio, pad = preprocess(img)
    
    cfx.push()
    try:
        d_input = cuda.mem_alloc(input_image.nbytes)
        cuda.memcpy_htod(d_input, input_image)
        
        num_classes = len(LABELS)
        output_len = num_classes + 5 
        OUTPUT_SHAPE = (1, 25200, output_len) 
        
        output = np.empty(OUTPUT_SHAPE, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        bindings = [int(d_input), int(d_output)]
        context.execute_v2(bindings)

        cuda.memcpy_dtoh(output, d_output)
        d_input.free()
        d_output.free()
        
    except Exception as e:
        st.error(f"推論錯誤: {e}")
        return None, None, None
    finally:
        cfx.pop()
        
    return output[0], ratio, pad

def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0: return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]
    
    selected = []
    while len(idxs) > 0:
        current = idxs[0]
        selected.append(current)
        if len(idxs) == 1: break
        
        xx1 = np.maximum(x1[current], x1[idxs[1:]])
        yy1 = np.maximum(y1[current], y1[idxs[1:]])
        xx2 = np.minimum(x2[current], x2[idxs[1:]])
        yy2 = np.minimum(y2[current], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[current] + areas[idxs[1:]] - inter + 1e-6)
        idxs = idxs[1:][iou < iou_threshold]
        
    return selected
    
def postprocess(pred, ratio, pad, conf_thres=0.1): 
    if pred is None: return [], [], []
    
    boxes = []
    scores = []
    class_ids = []
    
    pred = pred[pred[:, 4] >= conf_thres]
    
    for det in pred:
        cls_scores = det[5:]
        cls_id = np.argmax(cls_scores)
        score = cls_scores[cls_id] * det[4]
        
        if score < conf_thres: continue
        
        cx, cy, w, h = det[:4]
        
        # 座標還原
        cx = (cx - pad[0])
        cy = (cy - pad[1])
        cx /= ratio[0]
        cy /= ratio[1]
        w /= ratio[0]
        h /= ratio[1]
        
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(cls_id)

    if not boxes: return [], [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    keep = nms(boxes, scores)
    return boxes[keep], scores[keep], class_ids[keep]

@st.cache_data(show_spinner=False)
def generate_recipe_gemini(items):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash') 
        
        # 👇 這裡的 Prompt 是關鍵，我們加強了限制指令
        prompt = (
            f"你是一位擅長『清冰箱料理』的台灣廚師。我現在只有這些食材：{', '.join(items)}。\n"
            "請嚴格遵守以下規則設計食譜：\n"
            "1. **請設計 3 道完全不同的料理**，風格可以包含快炒、湯品、涼拌或創意料理。\n"
            "2. **主要食材只能使用上述清單內的項目**，絕對禁止添加清單以外的肉類或蔬菜。\n"
            "3. 允許使用家中常備調味料（油、鹽、糖、醬油、米酒、醋、胡椒、水）。\n"
            "4. 每道食譜之間，請務必使用『|||』這三個直線符號作為分隔線，不要有其他多餘文字。\n"
            "\n"
            "請用**繁體中文**回答，每一道食譜的格式如下：\n"
            "### 菜名：[料理名稱]\n"
            "**食材：**\n[條列式清單]\n"
            "**調味料：**\n[列出需要的調味料]\n"
            "**步驟：**\n[條列式步驟]\n"
            "|||\n"
            "### 菜名：...\n"
            "(依此類推)"
        )
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        if "429" in str(e):
            return "⚠️ 生成太快了，Google 叫我們休息一下。請等待 30 秒後再試一次。"
        return f"食譜生成失敗：{str(e)}"

st.set_page_config(page_title="Jetson AI 廚房", layout="wide")
st.title("🍳 AI 智慧冰箱 (Jetson Nano/Orin + Gemini)")

st.write("---")
st.write("📸 **請上傳一張包含食材的照片 (支援 jpg, png)**")

uploaded = st.file_uploader("選擇照片...", type=["jpg", "png", "jpeg"])

if uploaded:
    img_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="原始圖片", channels="BGR", width="stretch")

    with st.spinner("AI 正在辨識食材中..."):
        pred, ratio, pad = infer(img) 
        boxes, scores, ids = postprocess(pred, ratio, pad)

    detected_items = set()
    img_draw = img.copy()

    if len(boxes) > 0:
        for box, score, cls in zip(boxes, scores, ids):
            x1, y1, x2, y2 = box
            
            if cls < len(LABELS):
                label_name = LABELS[cls]
            else:
                label_name = f"Unknown({cls})"
                
            detected_items.add(label_name)
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_draw, f"{label_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        st.info("畫面中未偵測到指定的食材。")

    with col2:
        st.image(img_draw, caption="AI 辨識結果", channels="BGR", width="stretch")

    st.divider()
    if detected_items:
        items_list = sorted(list(detected_items)) # 排序很重要，讓 Cache 知道這是同一組食材
        st.success(f"🧺 偵測到的食材：**{', '.join(items_list)}**")
        
        if API_KEY and "填在這裡" not in API_KEY:
            # 使用 form 來避免 Streamlit 重新整理時自動觸發
            with st.form("recipe_form"):
                submitted = st.form_submit_button("✨ 讓 Gemini 大廚設計食譜")
                
                if submitted:
                    with st.spinner("Gemini 正在為您設計三道精選料理..."):
                        # 呼叫生成函式
                        full_text = generate_recipe_gemini(items_list)
                        
                        # 處理錯誤訊息
                        if "食譜生成失敗" in full_text or "⚠️" in full_text:
                            st.error(full_text)
                        else:
                            # 使用設定好的分隔符號切割文字
                            recipes = full_text.split("|||")
                            
                            # 過濾掉可能的空白項目
                            recipes = [r.strip() for r in recipes if r.strip()]

                            # 建立分頁 (Tabs)
                            st.markdown("### 👨‍🍳 您的專屬食譜建議")
                            
                            # 檢查是否成功切成 3 份，若格式跑掉則顯示全部
                            if len(recipes) >= 2:
                                # 動態建立對應數量的 Tab
                                tabs = st.tabs([f"料理 {i+1}" for i in range(len(recipes))])
                                
                                for i, tab in enumerate(tabs):
                                    with tab:
                                        st.markdown(recipes[i])
                            else:
                                # 萬一 AI 沒乖乖加分隔線，就直接印出全部
                                st.markdown(full_text)
        else:
            st.error("⚠️ 請在程式碼開頭填入正確的 API Key！")