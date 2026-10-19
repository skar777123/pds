"""
Camera Calibration & Windshield Alignment Tool for Portronics USB Webcam.
Draws the upper 40% horizon crop guideline to help position the dashcam in the vehicle.
"""

import sys
import time
from pathlib import Path
import cv2

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, ROI_TOP_PIXELS, TARGET_FPS

def test_camera(src=CAMERA_INDEX):
    print("=" * 60)
    print("[CALIBRATION] Starting Portronics USB Webcam Alignment Tool")
    print(f"  Target Device: {src} | Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Horizon Mask: Top {ROI_TOP_PIXELS} px (Upper 40% will be ignored)")
    print("  Instructions: Tilt camera until horizon aligns with the RED line.")
    print("  Press 'q' to exit.")
    print("=" * 60)

    try:
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    except Exception:
        cap = cv2.VideoCapture(src)
        
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source {src}. Ensure camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame from camera...")
            time.sleep(0.05)
            continue

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time

        # Draw darkened overlay for top 40% (ignored zone)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, ROI_TOP_PIXELS), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw red horizon guide line
        cv2.line(frame, (0, ROI_TOP_PIXELS), (FRAME_WIDTH, ROI_TOP_PIXELS), (0, 0, 255), 3)
        cv2.putText(frame, "HORIZON ALIGNMENT LINE (DROP UPPER 40%)", (10, ROI_TOP_PIXELS - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw active road zone label
        cv2.putText(frame, "ACTIVE ROAD DETECTION ZONE (LOWER 60%)", (10, ROI_TOP_PIXELS + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # FPS indicator
        cv2.putText(frame, f"Portronics UVC Live @ {fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("Portronics Webcam Calibration - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[CALIBRATION] Alignment tool closed.")

if __name__ == "__main__":
    src_idx = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX
    test_camera(src=src_idx)
