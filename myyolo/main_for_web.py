# main_for_web.py

import asyncio
import cv2
import numpy as np
import time
import os
from pathlib import Path
import io  # Not strictly used in this version, but often useful
import httpx # 新增导入
import json # 用于加载和保存天气数据
from datetime import datetime, timedelta # 用于检查数据新鲜度

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse  # HTMLResponse, JSONResponse not used for core function here
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # For serving Vue frontend

# --- ONNX Runtime ---
import onnxruntime

# --- Configuration ---
ONNX_MODEL_PATH = 'best.onnx'  # Assumes best.onnx is in the same directory as this script
MODEL_INPUT_SIZE = (640, 640)
CLASS_NAMES = ['green-leafhopper', 'leaf-folder', 'rice-bug', 'stem-borer', 'whorl-maggot']
NUM_CLASSES = len(CLASS_NAMES)
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
REG_MAX = 16
CAMERA_INDEX = 0  # <<<--- IMPORTANT: Set this to your RK3588 USB camera index (e.g., 0, 1, or 21 if that's correct)
CAIYUN_API_TOKEN = "5SVW9mFqMU2lOE49"  # 彩云天气API Token
NANJING_LOCATION = "118.7674,32.0415"     # 南京市经纬度 (大约值，彩云API接受经纬度)

WEATHER_DATA_EXPIRY_HOURS = 24 # 数据有效期（小时）

# --- Global variables for managing camera and processing state ---
cap = None
is_processing_active = False
last_processed_frame_with_box_bytes = None  # Stores encoded JPEG bytes of the frame with box
last_high_confidence_box_info = None  # Stores (box_xyxy, class_name, confidence)
frame_lock = asyncio.Lock()  # Synchronizes access to shared frame data
current_detection_details = {  # 新增: 用于存储最新检测详情供API查询
    "name": "无",
    "confidence": 0.0,
    "box_xyxy": None  # [x1, y1, x2, y2]
}
current_weather_data = None
last_weather_update_time = None

# --- Log file paths ---
LOG_DIR = Path("./logs")  # 日志文件将存储在脚本同目录下的 logs 文件夹中
LOG_DIR.mkdir(exist_ok=True)  # 创建 logs 文件夹如果它不存在

X_ORIGIN_LOG_FILE = LOG_DIR / "x_origin.log"
Y_ORIGIN_LOG_FILE = LOG_DIR / "y_origin.log"
WEATHER_DATA_FILE = LOG_DIR / "weather_data.json" # 天气数据缓存文件

# --- ONNX Session (will be initialized in startup_event) ---
ort_session = None
input_name = None
output_names = []

# --- FastAPI App Instance ---
app = FastAPI(title="Insect Laser Killer Backend API")

# --- CORS Middleware (Good practice, especially if frontend might be on a different dev port initially) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, restrict to your frontend's domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions for ONNX Model ---
def _preprocess_image_onnx(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]
    model_w, model_h = MODEL_INPUT_SIZE
    scale = min(model_w / original_w, model_h / original_h)
    scaled_w, scaled_h = int(original_w * scale), int(original_h * scale)
    resized_img_rgb = cv2.resize(img_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    pad_w_offset = (model_w - scaled_w) // 2
    pad_h_offset = (model_h - scaled_h) // 2
    padded_img_rgb_3d_hwc = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    padded_img_rgb_3d_hwc[pad_h_offset:pad_h_offset + scaled_h, pad_w_offset:pad_w_offset + scaled_w] = resized_img_rgb
    img_chw = padded_img_rgb_3d_hwc.transpose(2, 0, 1)
    input_tensor_nchw_uint8 = np.expand_dims(img_chw, axis=0)
    input_tensor_nchw_float32 = input_tensor_nchw_uint8.astype(np.float32) / 255.0
    return np.ascontiguousarray(input_tensor_nchw_float32), scale, pad_w_offset, pad_h_offset


def _decode_dfl_box_onnx(pred_dist_single_box_raw, anchor_x, anchor_y, reg_max_val=REG_MAX):
    pred_dist_reshaped = pred_dist_single_box_raw.reshape(4, reg_max_val)
    pred_dist_softmax = np.zeros_like(pred_dist_reshaped, dtype=np.float32)
    for k_side in range(4):
        side_dist_logits = pred_dist_reshaped[k_side, :].astype(np.float32)
        max_logit = np.max(side_dist_logits)
        exp_dist = np.exp(side_dist_logits - max_logit)
        sum_exp_dist = np.sum(exp_dist)
        if sum_exp_dist > 1e-9:
            pred_dist_softmax[k_side, :] = exp_dist / sum_exp_dist
        else:
            pred_dist_softmax[k_side, :] = 1.0 / reg_max_val
    project_vector = np.arange(reg_max_val, dtype=np.float32).reshape(1, reg_max_val)
    ltrb_distances = np.sum(pred_dist_softmax * project_vector, axis=1)
    dl, dt, dr, db = ltrb_distances
    x1 = anchor_x - dl;
    y1 = anchor_y - dt;
    x2 = anchor_x + dr;
    y2 = anchor_y + db
    return [x1, y1, x2, y2]


def _postprocess_outputs_onnx(outputs_list, original_shape, scale_factor, pad_w_offset, pad_h_offset):
    if len(outputs_list) != 6: return [], [], []
    strides = [8, 16, 32]
    all_decoded_boxes_xyxy_model, all_confidences, all_class_ids = [], [], []
    for i in range(3):
        box_feat_raw = np.squeeze(outputs_list[i * 2])  # Already float32 from ONNX Runtime
        cls_score_feat_sigmoid = np.squeeze(outputs_list[i * 2 + 1])  # Already float32 and sigmoided
        stride = strides[i]
        feat_h, feat_w = cls_score_feat_sigmoid.shape[1], cls_score_feat_sigmoid.shape[2]
        anchor_points_x = (np.arange(feat_w, dtype=np.float32) + 0.5) * stride
        anchor_points_y = (np.arange(feat_h, dtype=np.float32) + 0.5) * stride
        box_feat_hwc = box_feat_raw.transpose(1, 2, 0)
        cls_score_feat_hwc_sigmoid = cls_score_feat_sigmoid.transpose(1, 2, 0)
        for r_idx in range(feat_h):
            for c_idx in range(feat_w):
                scores_vector = cls_score_feat_hwc_sigmoid[r_idx, c_idx, :]
                final_confidence = np.max(scores_vector)
                if final_confidence < CONF_THRESHOLD: continue
                class_id = np.argmax(scores_vector)
                dfl_raw_data_for_box = box_feat_hwc[r_idx, c_idx, :]
                anchor_x, anchor_y = anchor_points_x[c_idx], anchor_points_y[r_idx]
                box_xyxy_model = _decode_dfl_box_onnx(dfl_raw_data_for_box, anchor_x, anchor_y)
                all_decoded_boxes_xyxy_model.append(box_xyxy_model)
                all_confidences.append(final_confidence)
                all_class_ids.append(class_id)
    if not all_decoded_boxes_xyxy_model: return [], [], []
    nms_boxes_xywh_model = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in all_decoded_boxes_xyxy_model]
    indices = cv2.dnn.NMSBoxes(nms_boxes_xywh_model, all_confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    final_boxes_scaled_xyxy, final_confidences, final_class_ids = [], [], []
    if isinstance(indices, tuple) and len(indices) > 0:
        indices = indices[0]
    elif not isinstance(indices, list) and hasattr(indices, 'flatten'):
        indices = indices.flatten().tolist()
    else:
        indices = [] if not isinstance(indices, list) else indices
    original_h, original_w = original_shape
    for idx in indices:
        x_m, y_m, w_m, h_m = nms_boxes_xywh_model[idx]
        x_s_np, y_s_np = x_m - pad_w_offset, y_m - pad_h_offset
        x_o, y_o, w_o, h_o = x_s_np / scale_factor, y_s_np / scale_factor, w_m / scale_factor, h_m / scale_factor
        x1, y1, x2, y2 = max(0, int(x_o)), max(0, int(y_o)), min(original_w, int(x_o + w_o)), min(original_h,
                                                                                                  int(y_o + h_o))
        final_boxes_scaled_xyxy.append([x1, y1, x2, y2]);
        final_confidences.append(all_confidences[idx]);
        final_class_ids.append(all_class_ids[idx])
    return final_boxes_scaled_xyxy, final_confidences, final_class_ids


def _draw_single_highest_confidence_box(image_bgr, box_info):
    if not box_info: return image_bgr
    img_with_box = image_bgr.copy()
    box_xyxy, class_name, conf = box_info
    x1, y1, x2, y2 = map(int, box_xyxy)
    label = f"{class_name}: {conf:.2f}"
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker box
    cv2.putText(img_with_box, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img_with_box


# --- Background Task for Camera Processing ---
async def process_camera_frames():
    global cap, is_processing_active, last_processed_frame_with_box_bytes
    global last_high_confidence_box_info, current_detection_details, frame_lock

    frame_counter = 0
    capture_delay_seconds = 0.1
    process_every_n_captures = int(0.5 / capture_delay_seconds)

    print(
        f"Camera processing: capture delay {capture_delay_seconds * 1000}ms, process every {process_every_n_captures} captures.")

    while is_processing_active and cap and cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            print("Warning: Failed to grab frame from camera.")
            await asyncio.sleep(0.5);
            continue

        frame_counter += 1
        current_frame_to_display = frame_bgr.copy()

        # --- 重置本次处理的检测详情 ---
        # 这样如果本次没有检测到，API获取到的会是旧的或者默认值
        # 或者我们可以选择只在有新检测时更新 current_detection_details

        if frame_counter % process_every_n_captures == 0:
            print(f"Processing frame {frame_counter}...")
            input_data, scale, pad_w, pad_h = _preprocess_image_onnx(frame_bgr)
            outputs = ort_session.run(output_names, {input_name: input_data})
            boxes_xyxy, confs, cls_ids = _postprocess_outputs_onnx(outputs, frame_bgr.shape[:2], scale, pad_w, pad_h)

            new_detection_this_frame = False
            if boxes_xyxy:  # 如果本次处理有检测结果
                max_conf_idx = np.argmax(confs)
                h_box_xyxy = boxes_xyxy[max_conf_idx]  # [x1, y1, x2, y2] scaled to original frame
                h_conf = confs[max_conf_idx]
                h_cls_id = cls_ids[max_conf_idx]
                class_name = CLASS_NAMES[h_cls_id] if 0 <= h_cls_id < NUM_CLASSES else "Unknown"

                # 更新全局变量用于绘制和API
                last_high_confidence_box_info = (h_box_xyxy, class_name, h_conf)
                current_detection_details = {
                    "name": class_name,
                    "confidence": float(h_conf),  # 确保是Python float
                    "box_xyxy": [int(c) for c in h_box_xyxy]  # 确保是Python int列表
                }
                new_detection_this_frame = True
                print(f"  Detected: {class_name}, Conf: {h_conf:.2f}, Box: {h_box_xyxy}")

                # --- 写入日志文件 ---
                # 计算目标中心点 (使用原始检测框坐标)
                center_x = int(h_box_xyxy[0] + (h_box_xyxy[2] - h_box_xyxy[0]) / 2)
                center_y = int(h_box_xyxy[1] + (h_box_xyxy[3] - h_box_xyxy[1]) / 2)

                try:
                    # 'w'模式会覆盖文件，只保留最后一行。如果需要追加历史，用'a'。
                    # 这里根据需求“最后一行分别表示”，使用 'w'
                    with open(X_ORIGIN_LOG_FILE, 'w') as f_x:
                        f_x.write(str(center_x) + "\n")
                    with open(Y_ORIGIN_LOG_FILE, 'w') as f_y:
                        f_y.write(str(center_y) + "\n")
                except Exception as e:
                    print(f"Error writing to log files: {e}")

            # 如果这帧没有检测到任何物体，但之前有检测结果，我们不清空 last_high_confidence_box_info
            # 这样旧的框会继续显示。但 current_detection_details 可以反映“当前无新检测”
            if not new_detection_this_frame and is_processing_active:  # 检查 is_processing_active 避免停止时也重置
                # 如果希望在没有新检测时清除API显示，可以在这里重置 current_detection_details
                # current_detection_details = {"name": "无 (本轮未检出)", "confidence": 0.0, "box_xyxy": None}
                # 或者保持上一次的检测结果，取决于你的产品逻辑
                pass  # 当前逻辑：如果没有新检测，current_detection_details 保持上一次的值

        # 始终在当前实时帧上绘制最新的（或上一个）最高置信度框
        if last_high_confidence_box_info:
            current_frame_to_display = _draw_single_highest_confidence_box(current_frame_to_display,
                                                                           last_high_confidence_box_info)

        try:
            _, encoded_image = cv2.imencode('.jpg', current_frame_to_display, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = encoded_image.tobytes()
        except Exception as e:
            print(f"Error encoding frame: {e}");
            frame_bytes = b''

        async with frame_lock:
            last_processed_frame_with_box_bytes = frame_bytes

        await asyncio.sleep(capture_delay_seconds)

    print("Camera processing loop has stopped.")
    async with frame_lock:  # 清理状态
        last_processed_frame_with_box_bytes = None
        last_high_confidence_box_info = None
        current_detection_details = {"name": "已停止", "confidence": 0.0, "box_xyxy": None}


async def fetch_and_cache_weather_data():
    global current_weather_data, last_weather_update_time
    print("Attempting to fetch new weather data from Caiyun API...")
    if not CAIYUN_API_TOKEN or CAIYUN_API_TOKEN == "YOUR_CAIYUN_API_TOKEN":
        print("Error: Caiyun API token is not configured.")
        # 如果没有token，可以尝试加载旧数据或者返回错误
        if WEATHER_DATA_FILE.exists():
            try:
                with open(WEATHER_DATA_FILE, 'r') as f:
                    data = json.load(f)
                current_weather_data = data.get("data")  # 假设文件存的是完整响应
                last_weather_update_time = datetime.fromisoformat(data.get("timestamp"))
                print("Loaded weather data from local cache due to missing token.")
                return current_weather_data
            except Exception as e:
                print(f"Error loading cached weather data: {e}")
        return None  # 或者抛出异常

    api_url = f"https://api.caiyunapp.com/v2.6/{CAIYUN_API_TOKEN}/{NANJING_LOCATION}/realtime"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, params={"alert": "true"})  # alert参数可选
            response.raise_for_status()  # 如果请求失败 (4xx, 5xx), 会抛出异常
            data = response.json()

            if data.get("status") == "ok":
                current_weather_data = data.get("result", {}).get("realtime", {})
                last_weather_update_time = datetime.now()
                # 保存到本地文件
                with open(WEATHER_DATA_FILE, 'w') as f:
                    json.dump({"timestamp": last_weather_update_time.isoformat(), "data": current_weather_data}, f)
                print(f"Successfully fetched and cached weather data: {current_weather_data}")
                return current_weather_data
            else:
                print(f"Error from Caiyun API: {data.get('description', 'Unknown error')}")
                # 尝试加载旧数据
                if WEATHER_DATA_FILE.exists():
                    with open(WEATHER_DATA_FILE, 'r') as f:
                        old_data_cache = json.load(f)
                    current_weather_data = old_data_cache.get("data")
                    last_weather_update_time = datetime.fromisoformat(old_data_cache.get("timestamp"))
                    print(f"Using stale data due to API error. Last good data from {last_weather_update_time}")
                    return current_weather_data
                return None  # 或者抛出异常
    except httpx.HTTPStatusError as e:
        print(f"HTTP error fetching weather data: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"General error fetching weather data: {e}")
    return None  # 或者抛出异常，或者返回旧数据


async def get_weather_data():
    global current_weather_data, last_weather_update_time

    data_loaded_from_cache_this_run = False
    # 启动时或需要时尝试加载本地缓存
    if current_weather_data is None and WEATHER_DATA_FILE.exists():  # 只在内存中没有数据时加载
        try:
            with open(WEATHER_DATA_FILE, 'r') as f:
                cached_content = json.load(f)
            # cached_content 的结构是 {"timestamp": "...", "data": {...actual weather data...}}
            current_weather_data = cached_content.get("data")  # 我们需要的是 "data" 字段下的内容
            last_weather_update_time = datetime.fromisoformat(cached_content.get("timestamp"))
            data_loaded_from_cache_this_run = True
            print(f"Loaded initial weather data from cache. Timestamp: {last_weather_update_time}")
        except Exception as e:
            print(f"Error loading initial cached weather data from {WEATHER_DATA_FILE}: {e}")
            current_weather_data = None  # 确保如果加载失败，状态是清晰的
            last_weather_update_time = None

    # 检查数据是否过期或（因加载失败而）不存在
    needs_fetch = False
    if current_weather_data is None or not current_weather_data.get("skycon"):  # 检查核心字段skycon是否存在
        needs_fetch = True
        print("Weather data is missing or incomplete in memory. Will try to fetch.")
    elif last_weather_update_time is None or \
            (datetime.now() - last_weather_update_time) > timedelta(hours=WEATHER_DATA_EXPIRY_HOURS):
        needs_fetch = True
        print(f"Weather data in memory is expired (last update: {last_weather_update_time}). Will try to fetch.")

    if needs_fetch:
        print("Attempting to fetch new data from Caiyun API...")
        # fetch_and_cache_weather_data 会更新全局的 current_weather_data 和 last_weather_update_time
        success = await fetch_and_cache_weather_data()
        if not success and not data_loaded_from_cache_this_run and not current_weather_data:  # 如果获取失败且之前没有成功从缓存加载
            print("Failed to fetch new weather data and no valid cache was loaded previously.")
            return {"error": "Failed to fetch new weather data and no cache available.", "skycon": None}
        elif not success and current_weather_data:
            print("Failed to fetch new data, using previously loaded/cached data if available.")
        elif success:
            print("Successfully fetched new data from API.")

    # --- 确保返回给前端的结构是正确的 ---
    if current_weather_data and current_weather_data.get("skycon"):
        # 这里的 current_weather_data 已经是实际的天气字典了 (即日志中 "data" 字段的内容)
        return {
            "temperature": current_weather_data.get("temperature"),
            "humidity": current_weather_data.get("humidity") * 100 if current_weather_data.get(
                "humidity") is not None else None,
            "skycon": current_weather_data.get("skycon"),
            "cloudrate": current_weather_data.get("cloudrate"),
            "description_api_status": current_weather_data.get("status"),  # 彩云API的status字段, 例如 "ok"
            "last_updated": last_weather_update_time.isoformat() if last_weather_update_time else None,
            # 你可以添加更多从 current_weather_data 中提取的字段
            "wind_speed": current_weather_data.get("wind", {}).get("speed"),
            "aqi_chn": current_weather_data.get("air_quality", {}).get("aqi", {}).get("chn")
        }
    elif current_weather_data and not current_weather_data.get("skycon"):
        print(f"Weather data in memory is incomplete (missing skycon): {current_weather_data}")
        return {"error": "Weather data in memory is incomplete.", "skycon": None}
    else:
        print("No weather data available in memory to return.")
        return {"error": "No weather data available.", "skycon": None}

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    global ort_session, input_name, output_names
    try:
        print("Initializing ONNX Runtime session at startup...")
        providers = ['CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        input_name = ort_session.get_inputs()[0].name
        output_names = [output.name for output in ort_session.get_outputs()]
        print(f"ONNX session initialized. Input: '{input_name}', Outputs: {output_names}")
    except Exception as e:
        print(f"FATAL: Could not initialize ONNX session at startup: {e}")
        ort_session = None  # Ensure it's None if init fails

    print("Attempting initial weather data fetch...")
    await get_weather_data() # 尝试在启动时加载/获取天气数据

@app.on_event("shutdown")
async def shutdown_event():
    global cap, is_processing_active
    print("Application shutting down...")
    is_processing_active = False
    if cap:
        print("Releasing camera capture...")
        cap.release()
    print("Shutdown complete.")


# --- API Endpoints ---
@app.post("/laser/start_processing", summary="Start camera processing and YOLO inference")
async def start_processing_endpoint(background_tasks: BackgroundTasks):
    global cap, is_processing_active, last_processed_frame_with_box_bytes, last_high_confidence_box_info
    if ort_session is None:
        raise HTTPException(status_code=500, detail="ONNX Model not loaded. Cannot start processing.")
    if is_processing_active:
        return {"status": "warning", "message": "Processing is already active."}

    print(f"Attempting to open camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = None
        print(f"Error: Could not open camera at index {CAMERA_INDEX}")
        raise HTTPException(status_code=500, detail=f"Could not open camera at index {CAMERA_INDEX}")

    # Optional: Set camera properties if needed, e.g., resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    is_processing_active = True
    async with frame_lock:  # Clear previous state on start
        last_processed_frame_with_box_bytes = None
        last_high_confidence_box_info = None
    background_tasks.add_task(process_camera_frames)
    print(f"Camera {CAMERA_INDEX} opened and processing started in background.")
    return {"status": "success", "message": "Camera processing started."}


@app.post("/laser/stop_processing", summary="Stop camera processing and clear detection data")
async def stop_processing_endpoint():
    global cap, is_processing_active, last_processed_frame_with_box_bytes
    global last_high_confidence_box_info, current_detection_details  # 确保 current_detection_details 也被操作

    if not is_processing_active and not cap:
        # 如果处理已经停止，或者摄像头从未启动，则直接返回成功
        print("Stop request received, but processing was not active or camera not open.")
        # 确保状态一致性
        is_processing_active = False  # 确保标志位正确
        async with frame_lock:
            last_processed_frame_with_box_bytes = None
            last_high_confidence_box_info = None
            current_detection_details = {"name": "已停止", "confidence": 0.0, "box_xyxy": None}
        return {"status": "success", "message": "Processing was not active or already stopped."}

    print("Received request to stop camera processing...")
    is_processing_active = False  # 1. Signal the background processing loop to stop

    # 2. Wait a short moment for the loop to detect the flag and potentially finish its current iteration.
    # The duration might depend on how long one iteration of `process_camera_frames` takes,
    # especially the `await asyncio.sleep(capture_delay_seconds)` part.
    # Waiting for slightly longer than one capture_delay_seconds cycle is usually sufficient.
    # If process_camera_frames has longer sleeps or blocking calls, this might need adjustment,
    # or a more robust signaling mechanism like asyncio.Event.
    await asyncio.sleep(max(0.2, 0.25))  # 给与2.5倍捕获延迟的时间退出

    # 3. Release camera resources
    if cap is not None:  # 再次检查 cap 是否存在，以防在等待期间被其他逻辑置None
        print("Releasing camera hardware...")
        cap.release()
        cap = None  # Set to None after releasing
        print("Camera released.")
    else:
        print("Camera was already None or released.")

    # 4. Safely clear shared data to stop video feed (show black) and reset detection info
    async with frame_lock:
        print("Clearing shared frame data and detection details...")
        last_processed_frame_with_box_bytes = None
        last_high_confidence_box_info = None
        current_detection_details = {"name": "已停止", "confidence": 0.0, "box_xyxy": None}

    print("Camera processing successfully stopped. Frame data and detection details cleared.")
    return {"status": "success", "message": "Camera processing stopped. Feed will show black."}


async def generate_video_stream():
    global last_processed_frame_with_box_bytes, frame_lock, is_processing_active

    # Create a reusable black frame
    black_frame_bytes = b''
    try:
        black_img = np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype=np.uint8)
        ret_val, encoded_black_frame = cv2.imencode('.jpg', black_img)
        if ret_val: black_frame_bytes = encoded_black_frame.tobytes()
    except Exception as e:
        print(f"Error creating black frame for stream: {e}")

    while True:
        await asyncio.sleep(0.05)  # Stream at ~20 FPS target
        current_bytes_to_send = black_frame_bytes

        if is_processing_active:
            async with frame_lock:
                if last_processed_frame_with_box_bytes:
                    current_bytes_to_send = last_processed_frame_with_box_bytes

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + current_bytes_to_send + b'\r\n')


@app.get("/laser/video_feed", summary="Get MJPEG video stream")
async def video_feed_endpoint():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/laser/trigger_action", summary="Simulate triggering laser action")
async def trigger_laser_action():
    global last_high_confidence_box_info
    target_info_str = "No valid target identified for laser."
    action_taken = False
    if last_high_confidence_box_info:
        box, name, conf = last_high_confidence_box_info
        target_info_str = f"Target: {name} (Confidence: {conf:.2f}) at Box: {box}"
        print(f"SIMULATING LASER ACTION ON: {target_info_str}")
        # In a real application, you would add your hardware control code here
        # For example:
        # center_x = box[0] + (box[2] - box[0]) / 2
        # center_y = box[1] + (box[3] - box[1]) / 2
        # result = control_laser_hardware(center_x, center_y)
        # if result: action_taken = True
        action_taken = True  # Simulate success

    if action_taken:
        return {"status": "success", "message": "Laser action triggered (simulated).", "target_info": target_info_str}
    else:
        return {"status": "warning", "message": "Laser action not taken (no target or simulation).",
                "target_info": target_info_str}


# --- Placeholder Endpoints for Other Functionalities ---
@app.get("/status_panel", summary="Get system status including weather")
async def get_status_panel_data_endpoint():
    weather = await get_weather_data()
    # 系统信息部分被删除
    return {
        "weather": weather
        # "system_info" 字段已移除
    }

# @app.get("/encyclopedia/{insect_name}", summary="Get encyclopedia entry for an insect (placeholder)")
# async def get_encyclopedia_entry(insect_name: str):
#     # This should connect to a database or a file with insect information
#     sample_data = {
#         "green-leafhopper": {"description": "The green leafhopper is a common agricultural pest...",
#                              "control_methods": "Use appropriate insecticides, biological control."},
#         "leaf-folder": {"description": "The leaf folder caterpillar rolls rice leaves...",
#                         "control_methods": "Monitor fields, pheromone traps."},
#         "rice-bug": {"description": "Rice bugs feed on developing rice grains...",
#                      "control_methods": "Timing of planting, biological agents."},
#         "stem-borer": {"description": "Stem borers are larvae of moths that bore into plant stems...",
#                        "control_methods": "Resistant varieties, insecticides."},
#         "whorl-maggot": {"description": "Whorl maggots damage young rice plants...",
#                          "control_methods": "Water management, early detection."}
#     }
#     if insect_name in sample_data:
#         return sample_data[insect_name]
#     raise HTTPException(status_code=404, detail=f"Encyclopedia entry for '{insect_name}' not found.")

@app.get("/laser/current_detection", summary="Get details of the latest highest confidence detection")
async def get_current_detection_details():
    # is_processing_active 检查是可选的，取决于是否希望在停止时仍能获取最后信息
    # if not is_processing_active and current_detection_details["name"] == "已停止":
    #     return {"name": "已停止", "confidence": 0.0, "box_xyxy": None}
    return current_detection_details


# --- Serve Static Frontend Files ---
# This assumes 'dist' directory is in the same directory as 'main_for_web.py'
# Adjust path if your directory structure is different.
# Example: If main_for_web.py is in /home/elf/myyolo/ and dist is /home/elf/myyolo/dist/
app.mount("/", StaticFiles(directory="./dist", html=True), name="static-frontend")

if __name__ == "__main__":
    import uvicorn

    print(f"--- Insect Laser Killer Backend ---")
    print(f"Script CWD: {os.getcwd()}")
    print(f"Serving static files from: {os.path.abspath('./dist')}")
    print(f"ONNX Model: {ONNX_MODEL_PATH}")
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Target Processing: 1 frame every 4 seconds (approx)")
    print(f"Video Stream available at /laser/video_feed")
    print(f"---------------------------------")
    uvicorn.run(app, host="0.0.0.0", port=8000)
