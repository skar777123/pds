"""
Training and Export Pipeline for Pothole Detection Model (YOLOv8n).
Fine-tunes YOLOv8n on custom road surface taxonomy and exports ONNX/TFLite artifacts for Raspberry Pi.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def train_and_export(epochs=15, imgsz=320, batch=8):
    base_dir = Path(__file__).resolve().parent.parent
    data_yaml = base_dir / "dataset" / "data.yaml"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Training] Initializing YOLOv8n backbone...")
    model = YOLO("yolov8n.pt")

    print(f"[Training] Starting fine-tuning for {epochs} epochs at resolution {imgsz}x{imgsz}...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2,
        mosaic=1.0,
        mixup=0.1,
        perspective=0.001,
        project=str(models_dir / "runs"),
        name="pothole_yolo",
        exist_ok=True,
        verbose=True
    )

    # Copy best PyTorch weights to models/
    best_pt = models_dir / "runs" / "pothole_yolo" / "weights" / "best.pt"
    target_pt = models_dir / "pothole_yolov8n.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        print(f"[Export] Saved best PyTorch model to: {target_pt}")
    else:
        # Fallback to last.pt
        last_pt = models_dir / "runs" / "pothole_yolo" / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, target_pt)
            print(f"[Export] Saved PyTorch model to: {target_pt}")

    # Export to ONNX (Optimized for ARM Cortex-A72 CPU, ONNXRuntime & OpenCV DNN)
    print(f"[Export] Exporting to ONNX format (opset 12, {imgsz}x{imgsz})...")
    trained_model = YOLO(str(target_pt))
    onnx_path = trained_model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=12)
    
    target_onnx = models_dir / "pothole_yolov8n.onnx"
    if Path(onnx_path).exists() and Path(onnx_path).resolve() != target_onnx.resolve():
        shutil.copy2(onnx_path, target_onnx)
    print(f"[Export] Saved ONNX model to: {target_onnx}")

    # Attempt TFLite export
    try:
        print(f"[Export] Attempting TFLite export...")
        tflite_path = trained_model.export(format="tflite", imgsz=imgsz, int8=False)
        target_tflite = models_dir / "pothole_yolov8n_int8.tflite"
        if Path(tflite_path).exists():
            shutil.copy2(tflite_path, target_tflite)
            print(f"[Export] Saved TFLite model to: {target_tflite}")
    except Exception as e:
        print(f"[Export] TFLite direct export skipped ({e}). ONNX is fully supported via OpenCV DNN and ONNXRuntime.")

    print("\n✅ [Training Complete] Production model artifacts ready in 'models/' directory.")

if __name__ == "__main__":
    train_and_export(epochs=12, imgsz=320, batch=8)
