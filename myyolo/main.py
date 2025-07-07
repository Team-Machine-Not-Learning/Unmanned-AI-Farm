import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import warnings
from pathlib import Path
import os
import sys
import cv2
import numpy as np
import time
# import threading # Not strictly needed for this frame skipping method

from PIL import Image, ImageTk

import onnxruntime

warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Less relevant on Linux

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# --- Configuration ---
ONNX_MODEL_PATH = 'best.onnx'
MODEL_INPUT_SIZE = (640, 640)
CLASS_NAMES = ['green-leafhopper', 'leaf-folder', 'rice-bug', 'stem-borer', 'whorl-maggot']
NUM_CLASSES = len(CLASS_NAMES)
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
REG_MAX = 16
CAMERA_INDEX = 21  # <<<--- USB camera on Linux (RK3588). Adjust!!!


class Controller:
    ui: 'Win'

    def __init__(self):
        self.cap = None
        self.is_playing = False
        self.is_capturing = False
        self.output_size_w, self.output_size_h = MODEL_INPUT_SIZE
        self.conf_threshold = CONF_THRESHOLD
        self.nms_threshold = NMS_THRESHOLD

        # --- Settings for Method B (Frame Skipping) ---
        self.delay = 30  # ms - Small delay for UI responsiveness and frequent frame capture
        # To achieve ~1 FPS processing, if camera is ~30 FPS, process 1 frame every 30.
        # If camera is ~15 FPS, process 1 frame every 15.
        # Adjust this based on your camera's typical output rate.
        self.process_every_n_frames = 30
        self.frame_counter = 0
        # --- End Settings for Method B ---

        self.detected_results_cache = {
            "boxes": [], "class_ids": [], "confidences": [],
            "image_for_plot": None, "original_image_for_combox": None
        }

        self.ort_session = None
        self.input_name = None
        self.output_names = None
        try:
            print(f"Loading ONNX model: {ONNX_MODEL_PATH}")
            providers = ['CPUExecutionProvider']
            print(f"ONNX Runtime: Using providers: {providers}")
            self.ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_names = [output.name for output in self.ort_session.get_outputs()]
            print(f"ONNX Model loaded. Input: '{self.input_name}', Outputs: {self.output_names}")
            if len(self.output_names) != 6:
                messagebox.showwarning("ONNX Warning", f"Expected 6 output tensors, got {len(self.output_names)}.")
        except Exception as e:
            messagebox.showerror("Error", f"ONNX Model Initialization failed: {e}")
            self.ort_session = None;
            return

    def init(self, ui):
        self.ui = ui
        self.ui.tk_button_close.config(state=tk.DISABLED)
        if not self.ort_session:
            for btn in [self.ui.tk_button_upload_image, self.ui.tk_button_video, self.ui.tk_button_cam]:
                btn.config(state=tk.DISABLED)
            messagebox.showinfo("Model Error", "ONNX model load failed. Functionality limited.")

    def _preprocess_image(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]
        model_w, model_h = self.output_size_w, self.output_size_h
        scale = min(model_w / original_w, model_h / original_h)
        scaled_w, scaled_h = int(original_w * scale), int(original_h * scale)
        resized_img_rgb = cv2.resize(img_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        pad_w_offset = (model_w - scaled_w) // 2
        pad_h_offset = (model_h - scaled_h) // 2
        padded_img_rgb_3d_hwc = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
        padded_img_rgb_3d_hwc[pad_h_offset:pad_h_offset + scaled_h,
        pad_w_offset:pad_w_offset + scaled_w] = resized_img_rgb
        img_chw = padded_img_rgb_3d_hwc.transpose(2, 0, 1)
        input_tensor_nchw_uint8 = np.expand_dims(img_chw, axis=0)
        input_tensor_nchw_float32 = input_tensor_nchw_uint8.astype(np.float32) / 255.0
        return np.ascontiguousarray(input_tensor_nchw_float32), scale, pad_w_offset, pad_h_offset

    def _decode_dfl_box(self, pred_dist_single_box_raw, anchor_x, anchor_y, reg_max_val=REG_MAX):
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

    def _postprocess_outputs(self, outputs_list, original_shape, scale_factor, pad_w_offset, pad_h_offset):
        if len(outputs_list) != 6: return [], [], []
        strides = [8, 16, 32]
        all_decoded_boxes_xyxy_model, all_confidences, all_class_ids = [], [], []
        for i in range(3):
            box_feat_raw = np.squeeze(outputs_list[i * 2])
            cls_score_feat_sigmoid = np.squeeze(outputs_list[i * 2 + 1])
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
                    if final_confidence < self.conf_threshold: continue
                    class_id = np.argmax(scores_vector)
                    dfl_raw_data_for_box = box_feat_hwc[r_idx, c_idx, :]
                    anchor_x, anchor_y = anchor_points_x[c_idx], anchor_points_y[r_idx]
                    box_xyxy_model = self._decode_dfl_box(dfl_raw_data_for_box, anchor_x, anchor_y)
                    all_decoded_boxes_xyxy_model.append(box_xyxy_model)
                    all_confidences.append(final_confidence)
                    all_class_ids.append(class_id)
        if not all_decoded_boxes_xyxy_model: return [], [], []
        nms_boxes_xywh_model = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in all_decoded_boxes_xyxy_model]
        indices = cv2.dnn.NMSBoxes(nms_boxes_xywh_model, all_confidences, self.conf_threshold, self.nms_threshold)
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

    def _draw_detections(self, image_bgr, boxes_xyxy, class_ids, confidences):
        img = image_bgr.copy()
        for i, b in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = map(int, b);
            c, id = confidences[i], class_ids[i];
            n = CLASS_NAMES[id] if 0 <= id < NUM_CLASSES else "Unk"
            lbl = f"{n}:{c:.2f}";
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2);
            cv2.putText(img, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img

    def display_image_on_gui(self, image_bgr_or_path, resize_to_display=True):
        if isinstance(image_bgr_or_path, str):
            if not os.path.exists(image_bgr_or_path):
                messagebox.showerror("Error", f"Bad path:{image_bgr_or_path}")
                try:
                    ph_img = Image.open("images/pre_show.png").resize((400, 400))
                    ph_photo = ImageTk.PhotoImage(ph_img)
                    self.ui.tk_label_show_image.config(image=ph_photo);
                    self.ui.tk_label_show_image.image = ph_photo
                except:
                    pass
                return
            img_pil = Image.open(image_bgr_or_path)
        else:
            img_pil = Image.fromarray(cv2.cvtColor(image_bgr_or_path, cv2.COLOR_BGR2RGB))

        if resize_to_display: img_pil = img_pil.resize((400, 400))
        ph = ImageTk.PhotoImage(img_pil)
        self.ui.tk_label_show_image.config(image=ph);
        self.ui.tk_label_show_image.image = ph

    def _update_ui_with_results(self, boxes, cls_ids, confs, time_s=None, is_processing=True):
        if not is_processing:  # If skipping frame, only update count if you want, or do nothing
            self.ui.tk_label_count_show.config(text="-")  # Indicate not processed
            self.ui.tk_label_time_show.config(text="-")
            # Optionally clear other fields or leave them from last processed frame
            # self.ui.tk_label_class_show.config(text=" "); ...
            return

        n = len(boxes);
        self.ui.tk_label_count_show.config(text=str(n));
        if time_s is not None: self.ui.tk_label_time_show.config(text=f'{time_s:.3f} s')
        self.ui.tk_select_box_select_obj_show['values'] = ("");
        self.ui.tk_select_box_select_obj_show.set('')
        for l in [self.ui.tk_label_class_show, self.ui.tk_label_conf_show, self.ui.tk_label_xmin_show,
                  self.ui.tk_label_ymin_show, self.ui.tk_label_xmax_show, self.ui.tk_label_ymax_show]: l.config(
            text=" ")
        self.detected_results_cache.update({"boxes": boxes, "class_ids": cls_ids, "confidences": confs})
        if n > 0:
            cl = ['全部'] + [f"{CLASS_NAMES[id]}_{i}" for i, id in enumerate(cls_ids) if 0 <= id < NUM_CLASSES]
            self.ui.tk_select_box_select_obj_show['values'] = tuple(cl)
            if cl: self.ui.tk_select_box_select_obj_show.current(0)
            id0 = cls_ids[0];
            self.ui.tk_label_class_show.config(text=CLASS_NAMES[id0] if 0 <= id0 < NUM_CLASSES else "Unk")
            self.ui.tk_label_conf_show.config(text=f"{confs[0] * 100:.2f} %")
            x1, y1, x2, y2 = map(int, boxes[0]);
            self.ui.tk_label_xmin_show.config(text=str(x1));
            self.ui.tk_label_ymin_show.config(text=str(y1));
            self.ui.tk_label_xmax_show.config(text=str(x2));
            self.ui.tk_label_ymax_show.config(text=str(y2))

    def upload_image(self, evt):
        if not self.ort_session: messagebox.showerror("E", "ONNX Model not loaded.");return
        self.close_video_cam(None);
        self.ui.tk_select_box_select_obj_show.config(state=tk.NORMAL)
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if not p: return
        img0 = cv2.imread(p)
        if img0 is None: messagebox.showerror("E", f"Failed read:{p}");return
        self.detected_results_cache["original_image_for_combox"] = img0.copy()
        tS = time.time();
        inp, sc, pW, pH = self._preprocess_image(img0)
        outs = self.ort_session.run(self.output_names, {self.input_name: inp})
        b, c, ids = self._postprocess_outputs(outs, img0.shape[:2], sc, pW, pH);
        tE = time.time()
        imgWB = self._draw_detections(img0, b, ids, c);
        self.detected_results_cache["image_for_plot"] = imgWB
        self.display_image_on_gui(imgWB);
        self._update_ui_with_results(b, ids, c, time_s=(tE - tS))

    def combox_change(self, evt):
        txt = self.ui.tk_select_box_select_obj_show.get()
        if not self.detected_results_cache["boxes"] or not txt: return
        img0 = self.detected_results_cache["original_image_for_combox"]
        if img0 is None: print("W: Orig img for combox redraw not found."); self.display_image_on_gui(
            self.detected_results_cache["image_for_plot"]); return
        bs, ids, cs = self.detected_results_cache["boxes"], self.detected_results_cache["class_ids"], \
        self.detected_results_cache["confidences"]
        if txt == '全部':
            if self.detected_results_cache["image_for_plot"] is not None: self.display_image_on_gui(
                self.detected_results_cache["image_for_plot"])
            if bs: id0 = ids[0];self.ui.tk_label_class_show.config(
                text=CLASS_NAMES[id0] if 0 <= id0 < NUM_CLASSES else "Unk");self.ui.tk_label_conf_show.config(
                text=f"{cs[0] * 100:.2f}%");x1, y1, x2, y2 = map(int, bs[0]);self.ui.tk_label_xmin_show.config(
                text=str(x1));self.ui.tk_label_ymin_show.config(text=str(y1));self.ui.tk_label_xmax_show.config(
                text=str(x2));self.ui.tk_label_ymax_show.config(text=str(y2))
        else:
            try:
                idx = int(txt.split('_')[-1])
                if 0 <= idx < len(bs):
                    imgS = img0.copy()
                    for i, b in enumerate(bs):
                        if i != idx: x1, y1, x2, y2 = map(int, b);cv2.rectangle(imgS, (x1, y1), (x2, y2),
                                                                                (100, 100, 100), 1)
                    sB, sID, sC = bs[idx], ids[idx], cs[idx];
                    x1, y1, x2, y2 = map(int, sB);
                    lbl = f"{CLASS_NAMES[sID]}:{sC:.2f}"
                    cv2.rectangle(imgS, (x1, y1), (x2, y2), (0, 0, 255), 3);
                    cv2.putText(imgS, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    self.display_image_on_gui(imgS)
                    self.ui.tk_label_class_show.config(text=CLASS_NAMES[sID] if 0 <= sID < NUM_CLASSES else "Unk");
                    self.ui.tk_label_conf_show.config(text=f"{sC * 100:.2f}%")
                    self.ui.tk_label_xmin_show.config(text=str(x1));
                    self.ui.tk_label_ymin_show.config(text=str(y1));
                    self.ui.tk_label_xmax_show.config(text=str(x2));
                    self.ui.tk_label_ymax_show.config(text=str(y2))
            except ValueError:
                print(f"W:Could not parse idx from combox:{txt}")

    def _process_frame_common(self, frame_bgr, perform_inference=True):
        self.detected_results_cache["original_image_for_combox"] = frame_bgr.copy()
        if perform_inference:
            tS = time.time();
            inp, sc, pW, pH = self._preprocess_image(frame_bgr)
            outs = self.ort_session.run(self.output_names, {self.input_name: inp})
            b, c, ids = self._postprocess_outputs(outs, frame_bgr.shape[:2], sc, pW, pH);
            tE = time.time()
            imgWB = self._draw_detections(frame_bgr, b, ids, c);
            self.detected_results_cache["image_for_plot"] = imgWB
            self.display_image_on_gui(imgWB);
            self._update_ui_with_results(b, ids, c, time_s=(tE - tS), is_processing=True)
        else:
            self.display_image_on_gui(frame_bgr)  # Display raw frame
            self._update_ui_with_results([], [], [], is_processing=False)  # Update UI to show it's not processed

    def upload_video(self, evt):
        if not self.ort_session: messagebox.showerror("E", "ONNX Model not loaded.");return
        self.close_video_cam(None);
        btns = [self.ui.tk_button_upload_image, self.ui.tk_button_video, self.ui.tk_button_cam];
        [b.config(state=tk.DISABLED) for b in btns]
        self.ui.tk_button_close.config(state=tk.NORMAL);
        self.ui.tk_select_box_select_obj_show.config(state=tk.DISABLED)
        vp = filedialog.askopenfilename(title="Select video", filetypes=[("Videos", "*.mp4;*.avi;*.mov")])
        if vp:
            self.cap = cv2.VideoCapture(vp)
            if not self.cap.isOpened(): messagebox.showerror("E", f"Could not open video:{vp}");self.close_video_cam(
                None);return
            self.is_playing = True;
            self.frame_counter = 0;
            self.update_video_frame_loop()

    def update_video_frame_loop(self):
        if self.is_playing and self.cap and self.cap.isOpened():
            ret, fb = self.cap.read()
            if ret:
                self.frame_counter += 1
                perform_inf = (self.frame_counter % self.process_every_n_frames == 0)
                self._process_frame_common(fb, perform_inference=perform_inf)
                self.ui.after(self.delay, self.update_video_frame_loop)
            else:
                self.close_video_cam(None)

    def open_cam(self, evt):
        if not self.ort_session: messagebox.showerror("E", "ONNX Model not loaded.");return
        self.close_video_cam(None);
        btns = [self.ui.tk_button_upload_image, self.ui.tk_button_video, self.ui.tk_button_cam];
        [b.config(state=tk.DISABLED) for b in btns]
        self.ui.tk_button_close.config(state=tk.NORMAL);
        self.ui.tk_select_box_select_obj_show.config(state=tk.DISABLED)
        if not self.is_capturing:
            print(f"Attempting to open camera index: {CAMERA_INDEX}")
            self.cap = cv2.VideoCapture(CAMERA_INDEX)  # No CAP_DSHOW for Linux
            if not self.cap.isOpened(): messagebox.showerror("E",
                                                             f"Could not open cam (idx {CAMERA_INDEX}).");self.close_video_cam(
                None);return
            print(f"Camera {CAMERA_INDEX} opened successfully.")
            self.is_capturing = True;
            self.frame_counter = 0;
            self.update_camera_frame_loop()

    def update_camera_frame_loop(self):
        if self.is_capturing and self.cap and self.cap.isOpened():
            ret, fb = self.cap.read()
            if ret:
                self.frame_counter += 1
                perform_inf = (self.frame_counter % self.process_every_n_frames == 0)
                self._process_frame_common(fb, perform_inference=perform_inf)
                self.ui.after(self.delay, self.update_camera_frame_loop)
            else:
                messagebox.showerror("Cam E", "Failed to capture frame.");self.close_video_cam(None)

    def close_video_cam(self, evt):
        self.is_playing = False;
        self.is_capturing = False
        if self.cap: self.cap.release();self.cap = None
        btns = [self.ui.tk_button_upload_image, self.ui.tk_button_video, self.ui.tk_button_cam];
        [b.config(state=tk.NORMAL) for b in btns]
        self.ui.tk_button_close.config(state=tk.DISABLED)
        try:
            self.display_image_on_gui("images/pre_show.png")
        except Exception as e:
            print(f"Err displaying placeholder:{e}")
        self._update_ui_with_results([], [], [], is_processing=True)  # Clear results as if processed

    def __del__(self):
        if self.cap: self.cap.release()
        # No explicit ort_session.release() needed, Python GC handles it.
        print("Controller for ONNX on RK3588 is being destroyed.")


class WinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_button_upload_image = self.__tk_button_upload_image(self)
        self.tk_button_video = self.__tk_button_video(self)
        self.tk_button_cam = self.__tk_button_cam(self)
        self.tk_label_show_image = self.__tk_label_show_image(self)
        self.tk_button_close = self.__tk_button_close(self)
        self.tk_label_time = self.__tk_label_time(self)
        self.tk_label_time_show = self.__tk_label_time_show(self)
        self.tk_label_count = self.__tk_label_count(self)
        self.tk_label_count_show = self.__tk_label_count_show(self)
        self.tk_label_conf = self.__tk_label_conf(self)
        self.tk_label_conf_show = self.__tk_label_conf_show(self)
        self.tk_label_class = self.__tk_label_class(self)
        self.tk_label_class_show = self.__tk_label_class_show(self)
        self.tk_label_xmin = self.__tk_label_xmin(self)
        self.tk_label_xmin_show = self.__tk_label_xmin_show(self)
        self.tk_label_ymin = self.__tk_label_ymin(self)
        self.tk_label_ymin_show = self.__tk_label_ymin_show(self)
        self.tk_label_xmax = self.__tk_label_xmax(self)
        self.tk_label_xmax_show = self.__tk_label_xmax_show(self)
        self.tk_label_ymax = self.__tk_label_ymax(self)
        self.tk_label_ymax_show = self.__tk_label_ymax_show(self)
        self.tk_label_select_obj = self.__tk_label_select_obj(self)
        self.tk_select_box_select_obj_show = self.__tk_select_box_select_obj_show(self)

    def __win(self):
        self.title("昆虫目标检测 (ONNX on RK3588 CPU)")
        width, height = 519, 619;
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f'{width}x{height}+{int((sw - width) / 2)}+{int((sh - height) / 2)}');
        self.resizable(width=False, height=False)

    def __tk_button_upload_image(self, p):
        b = tk.Button(p, text="图片检测", takefocus=False);b.place(x=62, y=29, width=79, height=30);return b

    def __tk_button_video(self, p):
        b = tk.Button(p, text="视频检测", takefocus=False);b.place(x=155, y=29, width=84, height=30);return b

    def __tk_button_cam(self, p):
        b = tk.Button(p, text="摄像头检测", takefocus=False);b.place(x=253, y=29, width=90, height=30);return b

    def __tk_label_show_image(self, p):
        try:
            img = Image.open("images/pre_show.png")
        except FileNotFoundError:
            img = Image.new('RGB', (400, 400), color='lightgray');from PIL import ImageDraw;d = ImageDraw.Draw(
                img);d.text((150, 180), "No Image", fill="black")
        img = img.resize((400, 400));
        ph = ImageTk.PhotoImage(img);
        l = tk.Label(p, anchor="center");
        l.place(x=68, y=73, width=400, height=400);
        l.config(image=ph);
        l.image = ph;
        return l

    def __tk_button_close(self, p):
        b = tk.Button(p, text="关闭检测", takefocus=False);b.place(x=364, y=29, width=112, height=30);return b

    def __tk_label_time(self, p):
        l = tk.Label(p, text="用时:", anchor="w");l.place(x=43, y=488, width=50, height=30);return l

    def __tk_label_time_show(self, p):
        l = tk.Label(p, text="0.000 s", anchor="w");l.place(x=85, y=488, width=70, height=30);return l

    def __tk_label_count(self, p):
        l = tk.Label(p, text="目标数:", anchor="w");l.place(x=162, y=488, width=70, height=30);return l

    def __tk_label_count_show(self, p):
        l = tk.Label(p, text="0", anchor="w");l.place(x=224, y=488, width=40, height=30);return l

    def __tk_label_conf(self, p):
        l = tk.Label(p, text="置信度:", anchor="w");l.place(x=41, y=515, width=60, height=30);return l

    def __tk_label_conf_show(self, p):
        l = tk.Label(p, text="0.00 %", anchor="w");l.place(x=93, y=515, width=74, height=30);return l

    def __tk_label_class(self, p):
        l = tk.Label(p, text="类别:", anchor="w");l.place(x=197, y=515, width=50, height=30);return l

    def __tk_label_class_show(self, p):
        l = tk.Label(p, text="N/A", anchor="w", wraplength=110);l.place(x=243, y=515, width=119, height=30);return l

    def __tk_label_xmin(self, p):
        l = tk.Label(p, text="xmin:", anchor="w");l.place(x=41, y=550, width=50, height=30);return l

    def __tk_label_xmin_show(self, p):
        l = tk.Label(p, text="0", anchor="w");l.place(x=81, y=550, width=50, height=30);return l

    def __tk_label_ymin(self, p):
        l = tk.Label(p, text="ymin:", anchor="w");l.place(x=146, y=550, width=50, height=30);return l

    def __tk_label_ymin_show(self, p):
        l = tk.Label(p, text="0", anchor="w");l.place(x=186, y=550, width=50, height=30);return l

    def __tk_label_xmax(self, p):
        l = tk.Label(p, text="xmax:", anchor="w");l.place(x=271, y=550, width=50, height=30);return l

    def __tk_label_xmax_show(self, p):
        l = tk.Label(p, text="0", anchor="w");l.place(x=316, y=550, width=50, height=30);return l

    def __tk_label_ymax(self, p):
        l = tk.Label(p, text="ymax:", anchor="w");l.place(x=385, y=550, width=50, height=30);return l

    def __tk_label_ymax_show(self, p):
        l = tk.Label(p, text="0", anchor="w");l.place(x=435, y=550, width=50, height=30);return l

    def __tk_label_select_obj(self, p):
        l = tk.Label(p, text="选择目标:", anchor="w");l.place(x=275, y=488, width=70, height=30);return l

    def __tk_select_box_select_obj_show(self, p):
        cb = ttk.Combobox(p, state="readonly", width=15);cb['values'] = ("");cb.place(x=347, y=488, width=140,
                                                                                      height=30);return cb


class Win(WinGUI):
    def __init__(self, controller: Controller):
        self.ctl = controller;
        super().__init__();
        self.__event_bind();
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def __event_bind(self):
        self.tk_button_upload_image.config(command=lambda: self.ctl.upload_image(None))
        self.tk_button_video.config(command=lambda: self.ctl.upload_video(None))
        self.tk_button_cam.config(command=lambda: self.ctl.open_cam(None))
        self.tk_button_close.config(command=lambda: self.ctl.close_video_cam(None))
        self.tk_select_box_select_obj_show.bind('<<ComboboxSelected>>', self.ctl.combox_change)

    def on_closing(self):
        if messagebox.askokcancel("退出", "确定要退出程序吗?"): self.ctl.close_video_cam(None); self.destroy()


if __name__ == "__main__":
    if not Path("images/pre_show.png").exists():
        print("Warning: images/pre_show.png not found. Placeholder will be created if GUI starts.")
        Path("images").mkdir(exist_ok=True)

    print("Starting ONNX application on RK3588 (CPU)...")
    controller = Controller()
    if controller.ort_session is None:
        print("Exiting: ONNX model load/init failure.")
        root_error = tk.Tk();
        root_error.title("严重错误")
        tk.Label(root_error, text="ONNX模型文件加载失败。\n程序无法启动。", padx=30, pady=30, font=("Arial", 12)).pack()
        tk.Button(root_error, text="关闭", command=root_error.destroy, width=10).pack(pady=10)
        root_error.mainloop()
    else:
        app = Win(controller)
        controller.init(app)
        app.mainloop()
    print("Application finished.")