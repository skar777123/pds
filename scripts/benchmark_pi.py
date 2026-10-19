"""
Performance Benchmarking Tool for Raspberry Pi 4B & Edge Pothole Detection.
Measures:
- Preprocessing latency (ROI crop + letterbox resize)
- Pure model inference latency
- Post-processing latency (NMS + Tracking)
- Sustained FPS and CPU temperature on ARM Cortex-A72
"""

import sys
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.config import (
    ROI_TOP_PIXELS, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, ONNX_MODEL_PATH
)
from edge.detect_edge import EdgePotholeDetector

def get_cpu_temp():
    """Retrieve CPU core temperature on Raspberry Pi OS via vcgencmd."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8")
        return out.strip().replace("temp=", "")
    except Exception:
        return "N/A (Non-RPi host)"

def run_benchmark(iterations=100):
    print("=" * 65)
    print("⚡ RASPBERRY PI 4B EDGE PERFORMANCE BENCHMARK")
    print(f"   Model: {ONNX_MODEL_PATH}")
    print(f"   Test Iterations: {iterations}")
    print(f"   Initial CPU Temp: {get_cpu_temp()}")
    print("=" * 65)

    detector = EdgePotholeDetector(model_path=ONNX_MODEL_PATH)

    # Synthetic mock 640x480 road frame
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    print("[Benchmark] Warming up CPU inference engine (10 passes)...")
    for _ in range(10):
        road_roi = mock_frame[ROI_TOP_PIXELS:, :]
        _ = detector.infer(road_roi)

    preprocess_times = []
    infer_times = []
    total_times = []

    print(f"[Benchmark] Benchmarking {iterations} iterations...")
    for i in range(iterations):
        t0 = time.perf_counter()
        
        # 1. Preprocess: Crop top 40%
        t_pre_start = time.perf_counter()
        road_roi = mock_frame[ROI_TOP_PIXELS:, :]
        blob_img = cv2.resize(road_roi, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        blob = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        t_pre_end = time.perf_counter()
        preprocess_times.append((t_pre_end - t_pre_start) * 1000)

        # 2. Pure Inference
        t_inf_start = time.perf_counter()
        if detector.backend == "onnxruntime":
            preds = detector.session.run(None, {detector.input_name: blob})[0]
        elif detector.backend == "opencv_dnn":
            detector.net.setInput(blob)
            preds = detector.net.forward()
        t_inf_end = time.perf_counter()
        infer_times.append((t_inf_end - t_inf_start) * 1000)

        # 3. Post-process
        _ = detector.infer(road_roi)
        t_end = time.perf_counter()
        total_times.append((t_end - t0) * 1000)

    avg_pre = np.mean(preprocess_times)
    avg_inf = np.mean(infer_times)
    avg_tot = np.mean(total_times)
    sustained_fps = 1000.0 / avg_tot

    print("\n" + "=" * 65)
    print("📊 BENCHMARK RESULTS")
    print(f"   Inference Backend:      {detector.backend}")
    print(f"   Average Preprocess:     {avg_pre:.2f} ms")
    print(f"   Average Pure Inference: {avg_inf:.2f} ms")
    print(f"   Average End-to-End:     {avg_tot:.2f} ms")
    print(f"   Sustained Speed:        {sustained_fps:.1f} FPS")
    print(f"   Final CPU Temp:         {get_cpu_temp()}")
    print("=" * 65)

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_benchmark(iterations=iters)
