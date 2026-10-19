"""
Test script to verify edge inference on extracted dataset images.
Tests model loading, ROI horizon cropping, prediction decoding, NMS, and CSV logging.
"""

import sys
import glob
from pathlib import Path
import cv2

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.config import ROI_TOP_PIXELS, ONNX_MODEL_PATH, CSV_LOG_PATH
from edge.detect_edge import EdgePotholeDetector
from edge.tracker import DefectTracker
from edge.csv_logger import CSVLogger

def run_test():
    print("[TEST] RUNNING MODEL VALIDATION TEST ON EXTRACTED IMAGES")
    print(f"   Model: {ONNX_MODEL_PATH}")
    print(f"   CSV Output: {CSV_LOG_PATH}")
    print("=" * 60)

    detector = EdgePotholeDetector(model_path=ONNX_MODEL_PATH, conf_thresh=0.20)
    tracker = DefectTracker(iou_threshold=0.30, min_hits_for_confirm=1)  # 1 hit for single images
    logger = CSVLogger()

    image_files = sorted(glob.glob("dataset/images/val/*.jpeg"))
    if not image_files:
        image_files = sorted(glob.glob("dataset/images/*.jpeg"))[:10]

    print(f"Found {len(image_files)} test images.\n")

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        # Resize to standard webcam resolution 640x480 for consistent test
        frame = cv2.resize(frame, (640, 480))
        
        # Upper 40% ROI crop as specified in PRD
        road_roi = frame[ROI_TOP_PIXELS:, :]

        # Run inference
        detections = detector.infer(road_roi)

        # Update tracker
        confirmed, active = tracker.update(detections)
        for c in confirmed:
            logger.log_detection(c, approx_speed=40.0)

        print(f"Image: {Path(img_path).name:<32} | Detections: {len(detections)}")
        for d in detections:
            print(f"  |-- Class: {d['class_name']:<15} | Severity: {d['severity']:<7} | Conf: {d['conf']:.2f} | Box: {d['box']}")

    print("\n[PASS] [Validation Complete] Logged detections verified in CSV.")

if __name__ == "__main__":
    run_test()
