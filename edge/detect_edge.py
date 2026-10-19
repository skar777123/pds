"""
Production Edge Inference Engine for Raspberry Pi 4B & Portronics USB Webcam.
Implements:
- 640x480 @ 30 FPS capture via threaded CameraStream
- ROI Horizon Cropping (upper 40% masked as per PRD Section 4)
- Accelerated Inference via OpenCV DNN / ONNX Runtime / TFLite
- Spatial-Temporal IoU Tracking & Deduplication
- Atomic Timestamped CSV Defect Logging
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Ensure edge module is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    ROI_CROP_TOP_RATIO, ROI_TOP_PIXELS, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT,
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    TFLITE_MODEL_PATH, ONNX_MODEL_PATH,
    CLASS_NAMES, SEVERITY_LEVELS
)
from edge.camera_stream import CameraStream
from edge.tracker import DefectTracker
from edge.csv_logger import CSVLogger

class EdgePotholeDetector:
    """Manages inference backends, frame preprocessing, and detection decoding."""
    
    def __init__(self, model_path=None, conf_thresh=CONFIDENCE_THRESHOLD, iou_thresh=NMS_IOU_THRESHOLD):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.backend = None
        self.net = None
        self.session = None
        self.interpreter = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        """Auto-detect and initialize the fastest available runtime backend on ARM/x86."""
        if model_path is not None and Path(model_path).exists():
            chosen_path = Path(model_path)
        elif ONNX_MODEL_PATH.exists():
            chosen_path = ONNX_MODEL_PATH
        elif TFLITE_MODEL_PATH.exists():
            chosen_path = TFLITE_MODEL_PATH
        else:
            # Check models/ for any .onnx or .pt
            models_dir = Path(__file__).resolve().parent.parent / "models"
            onnx_files = list(models_dir.glob("*.onnx"))
            if onnx_files:
                chosen_path = onnx_files[0]
            else:
                chosen_path = None

        if chosen_path is None or not chosen_path.exists():
            raise FileNotFoundError(f"[Detector] No valid model found in {MODEL_DIR}. Please run training first.")

        ext = chosen_path.suffix.lower()
        print(f"[Detector] Loading model artifact: {chosen_path} ({ext})")

        if ext == ".onnx":
            # Attempt ONNX Runtime with CPUExecutionProvider, fallback to OpenCV DNN
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(str(chosen_path), opts, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.backend = "onnxruntime"
                print("[Detector] Backend initialized: ONNXRuntime (ARM NEON / Multithreaded CPU)")
            except Exception as e:
                print(f"[Detector] ONNXRuntime unavailable ({e}). Falling back to OpenCV DNN.")
                self.net = cv2.dnn.readNetFromONNX(str(chosen_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "opencv_dnn"
                print("[Detector] Backend initialized: OpenCV DNN Engine")

        elif ext == ".tflite":
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    import tensorflow.lite as tflite
                    Interpreter = tflite.Interpreter

            self.interpreter = Interpreter(model_path=str(chosen_path), num_threads=4)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.backend = "tflite"
            print("[Detector] Backend initialized: TFLite Runtime with XNNPACK")

    def infer(self, road_roi):
        """
        Run forward pass on cropped road surface ROI.
        road_roi: BGR image slice (e.g. 640x288)
        Returns: list of detections [{'box': [x1, y1, x2, y2], 'class_id': int, 
                                     'class_name': str, 'severity': str, 'conf': float}]
        """
        h_roi, w_roi = road_roi.shape[:2]
        
        # 1. Letterbox resize to 320x320
        blob_img = cv2.resize(road_roi, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        blob = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)   # 1, 3, 320, 320

        # 2. Forward pass based on active backend
        if self.backend == "onnxruntime":
            preds = self.session.run(None, {self.input_name: blob})[0]  # shape: (1, 9, 2100) or (1, 2100, 9)
        elif self.backend == "opencv_dnn":
            self.net.setInput(blob)
            preds = self.net.forward()
        elif self.backend == "tflite":
            self.interpreter.set_tensor(self.input_details[0]['index'], blob)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Transpose if output is (1, 9, 2100) -> (2100, 9)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        # Output columns: [xc, yc, w, h, class0, class1, class2, class3, class4]
        boxes = []
        confidences = []
        class_ids = []

        scale_x = w_roi / MODEL_INPUT_WIDTH
        scale_y = h_roi / MODEL_INPUT_HEIGHT

        for row in preds:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if conf >= self.conf_thresh:
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                x1 = int((xc - w / 2.0) * scale_x)
                y1 = int((yc - h / 2.0) * scale_y)
                bw = int(w * scale_x)
                bh = int(h * scale_y)
                
                # Clip to ROI boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                bw = min(w_roi - x1, bw)
                bh = min(h_roi - y1, bh)

                if bw > 10 and bh > 10:
                    boxes.append([x1, y1, bw, bh])
                    confidences.append(conf)
                    class_ids.append(class_id)

        # 3. Apply Non-Maximum Suppression (NMS)
        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes[idx]
                    cid = class_ids[idx]
                    c_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"Defect_{cid}"
                    
                    # Convert ROI y-coordinates to Full Frame coordinates
                    full_y1 = by + ROI_TOP_PIXELS
                    full_y2 = full_y1 + bh
                    full_x1 = bx
                    full_x2 = bx + bw

                    # Estimate Severity heuristic based on defect size and class
                    rel_area = (bw * bh) / (w_roi * h_roi)
                    if cid == 1:  # Manhole
                        severity = "highest" if rel_area > 0.12 else ("high" if rel_area > 0.05 else "medium")
                    elif cid == 2:  # Water clogged
                        severity = "high" if rel_area > 0.08 else "medium"
                    else:
                        severity = "high" if rel_area > 0.10 else ("medium" if rel_area > 0.04 else "low")

                    detections.append({
                        "box": [full_x1, full_y1, full_x2, full_y2],
                        "class_id": cid,
                        "class_name": c_name,
                        "severity": severity,
                        "conf": confidences[idx]
                    })

        return detections

# Global buffer for web streaming mode
latest_detection_jpeg = None
detection_lock = threading.Lock()

class EdgeStreamingHandler(BaseHTTPRequestHandler):
    """HTTP MJPEG server for live browser viewing of pothole detections."""
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Raspberry Pi 4B - Pothole Detection Live</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ background: #0e0e10; color: #efeff1; font-family: sans-serif; text-align: center; margin: 0; padding: 15px; }}
        h2 {{ color: #00e676; margin-bottom: 5px; }}
        .stream-box {{ display: inline-block; border: 3px solid #00e676; border-radius: 8px; overflow: hidden; max-width: 100%; box-shadow: 0 4px 25px rgba(0,230,118,0.25); }}
        img {{ width: 100%; max-width: 640px; display: block; }}
        .hud {{ margin-top: 15px; font-family: monospace; font-size: 14px; background: #18181b; padding: 10px; border-radius: 6px; display: inline-block; }}
    </style>
</head>
<body>
    <h2>Raspberry Pi 4B Pothole Detection HUD</h2>
    <div class="stream-box">
        <img src="/stream.mjpg" alt="Live Defect Stream">
    </div>
    <br>
    <div class="hud">Active Pipeline: YOLOv8n (ARM NEON) | Logging: logs/pothole_detections.csv</div>
</body>
</html>"""
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with detection_lock:
                        if latest_detection_jpeg is None:
                            time.sleep(0.02)
                            continue
                        jpeg_bytes = latest_detection_jpeg
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg_bytes)))
                    self.end_headers()
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_edge_pipeline(model_path=None, video_src=CAMERA_INDEX, headless=False, web=False, port=8080):
    """Main execution loop for real-time edge pothole detection."""
    global latest_detection_jpeg

    # Auto-detect headless environment (SSH / no monitor)
    has_display = not (
        sys.platform.startswith("linux") and 
        "DISPLAY" not in os.environ and 
        "WAYLAND_DISPLAY" not in os.environ
    )

    if not headless and not web and not has_display:
        print("\n[INFO] No desktop display found ($DISPLAY is not set).")
        print("       Automatically operating in HEADLESS mode to prevent Qt GUI crash.")
        print("       (Tip: Pass --web to stream detections to your phone/laptop browser!)\n")
        headless = True

    print("=" * 65)
    print("[EDGE] STARTING EDGE POTHOLE DETECTION SYSTEM (Raspberry Pi 4B)")
    print(f"   Camera Source: {video_src} | Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"   ROI Crop: Dropping upper {int(ROI_CROP_TOP_RATIO * 100)}% ({ROI_TOP_PIXELS} px)")
    print(f"   Mode: {'Web Streamer' if web else ('Headless' if headless else 'Desktop GUI')}")
    print("=" * 65)

    detector = EdgePotholeDetector(model_path=model_path)
    tracker = DefectTracker(iou_threshold=0.35, max_lost=8, min_hits_for_confirm=2)
    logger = CSVLogger()
    camera = CameraStream(src=video_src, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=TARGET_FPS).start()

    if web:
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "localhost"

        server = HTTPServer(('0.0.0.0', port), EdgeStreamingHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"\n🌐 LIVE WEB DETECTION STREAMER ACTIVE!")
        print(f"   👉 Open in your phone/laptop browser: http://{local_ip}:{port}")
        print(f"   Press Ctrl+C to terminate.\n")

    fps_history = []
    total_detections_logged = 0

    try:
        while True:
            t_start = time.perf_counter()
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # 1. Apply ROI Masking (Drop upper 40% as specified in PRD Section 4)
            road_roi = frame[ROI_TOP_PIXELS:, :]

            # 2. Run Accelerated Model Inference
            raw_detections = detector.infer(road_roi)

            # 3. Spatial-Temporal Tracking & Deduplication
            confirmed_events, active_tracks = tracker.update(raw_detections)

            # 4. Atomic CSV Logging for newly confirmed defect events
            for event in confirmed_events:
                logger.log_detection(event, approx_speed=40.0)
                total_detections_logged += 1

            # 5. Calculate FPS
            t_elapsed = time.perf_counter() - t_start
            fps = 1.0 / t_elapsed if t_elapsed > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 6. Visualization & On-Screen Display (if GUI or Web Streamer is enabled)
            if not headless or web:
                # Draw ROI Horizon line
                cv2.line(frame, (0, ROI_TOP_PIXELS), (FRAME_WIDTH, ROI_TOP_PIXELS), (0, 0, 255), 2)
                cv2.putText(frame, "HORIZON MASK (UPPER 40% CROPPED)", (10, ROI_TOP_PIXELS - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                # Draw Active Tracked Defects
                for track in active_tracks:
                    bx1, by1, bx2, by2 = track.box
                    color_map = {
                        "low": (0, 255, 0),
                        "medium": (0, 255, 255),
                        "high": (0, 140, 255),
                        "highest": (0, 0, 255)
                    }
                    color = color_map.get(track.severity, (0, 255, 0))
                    
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    label = f"#{track.track_id} {track.class_name} [{track.severity.upper()}] {track.confidence:.2f}"
                    cv2.putText(frame, label, (bx1, max(20, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw Status HUD
                cv2.putText(frame, f"FPS: {avg_fps:.1f} | Backend: {detector.backend}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Verified Defects Logged: {total_detections_logged}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                if web:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    with detection_lock:
                        latest_detection_jpeg = jpeg.tobytes()

                if not headless and has_display:
                    try:
                        cv2.imshow("Raspberry Pi 4B Pothole Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("[Pipeline] Stop requested by user.")
                            break
                    except Exception:
                        headless = True

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")
    finally:
        camera.stop()
        if not headless and has_display:
            cv2.destroyAllWindows()
        print(f"[Pipeline] Finalized. Total logged road defects: {total_detections_logged}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi Pothole Detector")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX or TFLite model")
    parser.add_argument("--source", type=int, default=CAMERA_INDEX, help="Webcam device index (default: 0)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window for background daemon")
    parser.add_argument("--web", action="store_true", help="Stream live detections over HTTP to browser")
    parser.add_argument("--port", type=int, default=8080, help="Web stream port (default: 8080)")
    args = parser.parse_args()

    run_edge_pipeline(model_path=args.model, video_src=args.source, headless=args.headless, web=args.web, port=args.port)
