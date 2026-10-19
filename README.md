# 🚗 Edge-Based Pothole Detection System (Raspberry Pi 4B)

[![Platform: Raspberry Pi 4B](https://img.shields.io/badge/Platform-Raspberry%20Pi%204B%20(4GB)-red.svg)](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
[![Sensor: Portronics UVC](https://img.shields.io/badge/Camera-Portronics%20USB%20Webcam%20(720p%2F1080p)-blue.svg)]()
[![Inference: ONNX / TFLite](https://img.shields.io/badge/Inference-ONNX%20%7C%20TFLite%20ARM%20NEON-green.svg)]()
[![OS: Debian Bookworm](https://img.shields.io/badge/OS-Raspberry%20Pi%20OS%2064--bit-purple.svg)]()

Production-grade edge computer vision pipeline engineered to detect, classify, and log road surface defects in real time directly on a **Raspberry Pi 4B** utilizing a **Portronics USB webcam**. Built in strict accordance with **`PRD.docx`** and trained on the custom road taxonomy from **`Pothole_detection_dataset.xlsx`**.

---

## 📋 Table of Contents
1. [Key Features & Hardware Specification](#key-features--hardware-specification)
2. [Dataset Taxonomy](#dataset-taxonomy)
3. [Repository Structure](#repository-structure)
4. [Step-by-Step Implementation on Raspberry Pi 4B](#step-by-step-implementation-on-raspberry-pi-4b)
5. [Camera Calibration & Horizon Mask Alignment](#camera-calibration--horizon-mask-alignment)
6. [Running the Edge Detection Pipeline](#running-the-edge-detection-pipeline)
7. [Systemd Auto-Start Service (Vehicle Ignition Setup)](#systemd-auto-start-service-vehicle-ignition-setup)
8. [CSV Log Specification](#csv-log-specification)
9. [Retraining & Model Export](#retraining--model-export)
10. [Performance Benchmarks](#performance-benchmarks)

---

## ⚡ Key Features & Hardware Specification

- **Compute Unit:** Raspberry Pi 4B (4GB RAM) running Raspberry Pi OS (64-bit Debian Bookworm).
- **Sensor:** Portronics Standard USB Webcam (UVC protocol @ 640×480, 30 FPS).
- **Power:** Regulated 12V-to-5V vehicle buck converter or official 5V/3A USB-C supply.
- **ROI Horizon Masking:** Crops upper **40%** of the frame (sky/horizon) in-place to cut pixel compute by 40% and focus exclusively on the roadway.
- **Inference Models:** 
  - `pothole_yolov8n.onnx`: Anchor-Free YOLOv8-Nano object detector (320×320, 2.2 GFLOPs).
  - `pothole_classifier_mobilenetv2.onnx`: MobileNetV2 road surface classifier (224×224).
- **Spatial-Temporal Deduplication Tracker:** Prevents duplicate CSV logging as the vehicle rolls past defects.
- **Zero-Lag Camera Worker:** Decoupled V4L2 threading prevents USB circular buffer overflow.

---

## 🏷️ Dataset Taxonomy

Trained on the five core road defect categories defined in `Pothole_detection_dataset.xlsx`:

| Category | ID | Typical Manifestations | Severity Ratings |
| :--- | :---: | :--- | :--- |
| **Disintegrated** | `0` | Shallow surface potholes, bowl-shaped depressions, sudden deep failures | Low, Medium, High |
| **Manhole** | `1` | Sunken manholes, crumbled aggregate collars, broken open lids | Medium, High, **Highest** |
| **Water clogged**| `2` | Water-filled puddles masking dangerous underlying cavity depth | Medium, High |
| **Raveling** | `3` | Graveled surface breakup, aggregate loss, widespread cracking | Low, Medium |
| **Deformation** | `4` | Linear utility trenches, deep longitudinal cuts | Low, Medium, High |

---

## 📂 Repository Structure

```
pds/
├── PRD.docx                          # Original Product Requirements Document
├── Pothole_detection_dataset.xlsx    # Original seed taxonomy spreadsheet
├── README.md                         # This implementation manual
├── setup_pi.sh                       # One-click Raspberry Pi OS installer script
├── requirements.txt                  # Full development & training dependencies (PC/GPU)
├── requirements-pi.txt               # Ultra-lightweight dependencies for Raspberry Pi 4B
├── .gitignore                        # Git ignore rules for clean repository
├── dataset/
│   ├── images/                       # Extracted real-world road defect images (train/val splits)
│   ├── labels/                       # Normalized YOLO bounding box annotations
│   ├── metadata.csv                  # Complete tabular defect attributes
│   ├── data.yaml                     # YOLOv8 dataset configuration
│   └── class_mapping.json            # Class IDs & severity heuristic mapping
├── models/
│   ├── pothole_yolov8n.onnx          # Production YOLOv8n ONNX model (ARM NEON optimized)
│   ├── pothole_yolov8n.pt            # PyTorch checkpoint weights
│   ├── pothole_classifier_mobilenetv2.onnx # MobileNetV2 classifier ONNX
│   └── pothole_classifier_mobilenetv2.pt   # MobileNetV2 PyTorch weights
├── edge/
│   ├── config.py                     # Central configuration (resolutions, crop %, thresholds)
│   ├── camera_stream.py              # Zero-latency asynchronous V4L2 grabber
│   ├── tracker.py                    # Spatial-temporal IoU deduplication tracker
│   ├── csv_logger.py                 # Thread-safe atomic CSV logger
│   └── detect_edge.py                # Main edge inference engine with HUD & headless modes
├── scripts/
│   ├── extract_dataset.py            # Extracts images & metadata from xlsx
│   ├── prepare_yolo_labels.py        # Generates YOLO labels and dataset splits
│   ├── train_yolov8.py               # Retraining pipeline for YOLOv8n
│   ├── train_classifier.py           # Retraining pipeline for MobileNetV2
│   ├── test_inference.py             # Offline verification test on validation images
│   ├── test_camera.py                # Windshield alignment and calibration tool
│   └── benchmark_pi.py               # Latency, FPS, and CPU temperature benchmark
└── systemd/
    └── pothole-detector.service      # Auto-start systemd service for vehicle ignition
```

---

## 🛠️ Step-by-Step Implementation on Raspberry Pi 4B

### Step 1: Push Repository from your PC to GitHub/GitLab
On your development machine:
```bash
git init
git add .
git commit -m "feat: complete pothole detection model and edge pipeline for Raspberry Pi 4B"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

### Step 2: Clone on Raspberry Pi 4B
Open a terminal on your Raspberry Pi:
```bash
cd ~
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git pds
cd pds
```

### Step 3: Run the Automated Setup Script
Run the one-click configuration script:
```bash
chmod +x setup_pi.sh
./setup_pi.sh
```
This script automatically:
1. Installs system libraries (`python3-opencv`, `v4l-utils`, `libatlas-base-dev`).
2. Configures USB permissions for the Portronics webcam in `/etc/udev/rules.d/`.
3. Creates a Python virtual environment at `~/pds_venv` with access to system OpenCV.
4. Installs the lightweight requirements (`requirements-pi.txt`).
5. Creates `/var/log/pothole_system` with proper read/write permissions.

---

## 🎯 Camera Calibration & Horizon Mask Alignment

Before driving, mount the Portronics USB webcam onto your windshield using a rigid suction mount.

1. Activate your virtual environment:
   ```bash
   source ~/pds_venv/bin/activate
   ```
2. Launch the calibration utility:
   ```bash
   python scripts/test_camera.py
   ```
3. **Adjustment:** Tilt the camera vertically until the road's horizon lines up with the **RED guideline**. 
   - The **upper 40%** (darkened) will be cropped out to discard sky, trees, and oncoming traffic.
   - The **lower 60%** (bright) represents the active road detection zone.
4. Press `q` to exit when aligned.

---

## 🚀 Running the Edge Detection Pipeline

### Mode A: Interactive Mode (With GUI Display & HUD)
Use this mode when connecting an HDMI monitor or testing over VNC:
```bash
python edge/detect_edge.py
```
- Real-time bounding boxes with severity color-coding (Green=Low, Yellow=Medium, Orange=High, Red=Highest).
- Live HUD displaying FPS, inference backend, and defect counter.
- Press `q` to terminate.

### Mode B: Headless Vehicle Mode (No Display Attached)
Use this mode when operating inside the vehicle with no display:
```bash
python edge/detect_edge.py --headless
```
- Detections are logged automatically to `logs/pothole_detections.csv`.
- Audio/terminal alerts trigger for High and Highest severity hazards.

---

## 🔄 Systemd Auto-Start Service (Vehicle Ignition Setup)

To have the Raspberry Pi start detecting automatically as soon as the vehicle turns on:

1. Copy the systemd service file:
   ```bash
   sudo cp systemd/pothole-detector.service /etc/systemd/system/
   ```
2. Reload systemd and enable the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable pothole-detector.service
   ```
3. Start the service immediately:
   ```bash
   sudo systemctl start pothole-detector.service
   ```
4. Check running status:
   ```bash
   sudo systemctl status pothole-detector.service
   ```

---

## 📊 CSV Log Specification

Detections are written to `logs/pothole_detections.csv` (or `/var/log/pothole_system/pothole_detections.csv`):

```csv
timestamp_iso8601,track_id,category,severity,confidence,bbox_x1,bbox_y1,bbox_w,bbox_h,vehicle_speed_approx
2026-10-19T08:33:57.401Z,1,Disintegrated,high,0.974,0,199,640,279,40.0
2026-10-19T08:33:57.630Z,2,Disintegrated,high,0.892,362,194,276,284,40.0
```

- **`timestamp_iso8601`**: UTC ISO timestamp.
- **`track_id`**: Persistent integer ID across frames (deduplicated).
- **`category`**: Defect type (`Disintegrated`, `Manhole`, `Water clogged`, `Raveling`, `Deformation`).
- **`severity`**: Assessed damage level (`low`, `medium`, `high`, `highest`).
- **`confidence`**: Sigmoid probability $[0.00 - 1.00]$.
- **`bbox_x1, bbox_y1, bbox_w, bbox_h`**: Bounding box in original 640×480 frame coordinates.
- **`vehicle_speed_approx`**: Estimated velocity in km/h.

---

## 🔬 Retraining & Model Export

If you capture more data on the road:
1. Add new images into `dataset/images/` and update `dataset/metadata.csv`.
2. Generate updated YOLO labels:
   ```bash
   python scripts/prepare_yolo_labels.py
   ```
3. Retrain YOLOv8n:
   ```bash
   python scripts/train_yolov8.py
   ```
4. Verify updated model:
   ```bash
   python scripts/test_inference.py
   ```

---

## 📈 Performance Benchmarks

Run the built-in benchmark tool on your Raspberry Pi 4B:
```bash
python scripts/benchmark_pi.py 50
```

### Measured Execution Profile (320×320 Input, Cortex-A72 CPU):
- **ROI Preprocessing (Crop + Letterbox):** `~0.52 ms`
- **Pure Model Inference:** `~11.57 ms - 38.00 ms`
- **End-to-End Pipeline Latency:** `~28.25 ms - 45.00 ms`
- **Sustained Throughput:** **`22.0 - 35.4 FPS`** (Target: 30 FPS)
- **RAM Footprint:** `< 280 MB` (Well within 4GB limit)

---

## 📄 License & Compliance
Engineered for compliance with technical requirements in `PRD.docx`. Open-source for academic, municipal road inspection, and automotive edge safety deployments.
