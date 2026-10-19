"""
Edge System Configuration for Raspberry Pi 4B & Portronics Webcam Pothole Detection.
Strictly adheres to specifications in PRD.docx.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Hardware & Video Capture (PRD Section 2 & 4)
CAMERA_INDEX = 0                      # /dev/video0 on Raspberry Pi Linux
FRAME_WIDTH = 640                     # 640x480 resolution as specified in PRD
FRAME_HEIGHT = 480
TARGET_FPS = 30                       # 30 FPS target capture rate
FOURCC = "MJPG"                       # Portronics UVC hardware MJPEG compression

# Region of Interest (ROI) Masking (PRD Section 4)
# Crops upper 40% of the frame (sky/horizon) to focus computation exclusively on road surface
ROI_CROP_TOP_RATIO = 0.40             # Crop top 40% (0.0 to 1.0)
ROI_TOP_PIXELS = int(FRAME_HEIGHT * ROI_CROP_TOP_RATIO)  # 192 pixels cropped
ACTIVE_ROAD_HEIGHT = FRAME_HEIGHT - ROI_TOP_PIXELS        # 288 pixels active road

# Deep Learning Inference Engine
# Scaled to 320x320 for real-time ARM Cortex-A72 CPU execution
MODEL_INPUT_WIDTH = 320
MODEL_INPUT_HEIGHT = 320
NUM_CPU_THREADS = 4                   # Utilize all 4 Cortex-A72 cores

# Model Artifact Paths
TFLITE_MODEL_PATH = MODEL_DIR / "pothole_yolov8n_int8.tflite"
ONNX_MODEL_PATH = MODEL_DIR / "pothole_yolov8n.onnx"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "pothole_classifier_mobilenetv2.tflite"

# Detection & Tracking Thresholds
CONFIDENCE_THRESHOLD = 0.35           # Min confidence to consider detection
NMS_IOU_THRESHOLD = 0.45              # Non-Maximum Suppression IoU overlap limit
TRACKER_MAX_LOST_FRAMES = 10          # Frames before inactive track is deleted
MIN_HITS_FOR_ALERT = 2                # Consecutive detections to confirm physical defect

# Dataset Taxonomy (PRD Section 3 & Pothole_detection_dataset.xlsx)
CLASS_NAMES = [
    "Disintegrated",                  # 0: Shallow surface, bowl-shaped, deep structural potholes
    "Manhole",                        # 1: Sunken manholes, crumbled collars, broken open lids
    "Water clogged",                  # 2: Water-filled depressions masking hazard depth
    "Raveling",                       # 3: Shallow surface raveling, aggregate loss, cracks
    "Deformation"                     # 4: Trench potholes, deep longitudinal cuts
]

SEVERITY_LEVELS = ["low", "medium", "high", "highest"]

# Logging & Telemetry (PRD Section 4)
CSV_LOG_PATH = LOG_DIR / "pothole_detections.csv"
ENABLE_CONSOLE_LOG = True
ENABLE_SOUND_ALERT = True             # Audio beep / terminal alert for High/Highest severity
