"""
Camera Calibration & Windshield Alignment Tool for Portronics USB Webcam.
Draws the upper 40% horizon crop guideline to help position the dashcam in the vehicle.

Features:
- GUI Window mode (when HDMI monitor or VNC is connected)
- Auto-fallback Web Streamer (when running over SSH / headless)
- Snapshot mode (--snapshot to save image to disk)
"""

import os
import sys
import time
import socket
import argparse
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, ROI_TOP_PIXELS, TARGET_FPS

# Global frame buffer for web streaming
latest_jpeg = None
frame_lock = threading.Lock()

def get_local_ip():
    """Detect local IP address on LAN/Wi-Fi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP MJPEG video streaming handler for browser viewing over SSH."""
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Raspberry Pi Camera Calibration</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ background: #121212; color: #f0f0f0; font-family: sans-serif; text-align: center; margin: 0; padding: 15px; }}
        h2 {{ color: #ff5252; margin-bottom: 5px; }}
        p {{ color: #aaa; margin-top: 0; }}
        .stream-box {{ display: inline-block; border: 3px solid #ff5252; border-radius: 8px; overflow: hidden; max-width: 100%; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }}
        img {{ width: 100%; max-width: 640px; display: block; }}
        .badge {{ display: inline-block; background: #222; border: 1px solid #444; border-radius: 4px; padding: 6px 12px; margin-top: 15px; font-size: 14px; }}
    </style>
</head>
<body>
    <h2>Portronics Webcam Alignment</h2>
    <p>Tilt camera until road horizon matches the <b>RED GUIDELINE</b></p>
    <div class="stream-box">
        <img src="/stream.mjpg" alt="Live Camera Stream">
    </div>
    <br>
    <div class="badge">
        Upper 40% (Darkened): Cropped / Ignored &nbsp;|&nbsp; Lower 60%: Active Road
    </div>
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
                    with frame_lock:
                        if latest_jpeg is None:
                            time.sleep(0.02)
                            continue
                        jpeg_bytes = latest_jpeg

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
        # Suppress noisy HTTP request logging
        return

def annotate_frame(frame, fps=0.0):
    """Draw horizon line and active road regions."""
    # Darken upper 40%
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, ROI_TOP_PIXELS), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Red horizon guideline
    cv2.line(frame, (0, ROI_TOP_PIXELS), (FRAME_WIDTH, ROI_TOP_PIXELS), (0, 0, 255), 3)
    cv2.putText(frame, "HORIZON ALIGNMENT LINE (DROP UPPER 40%)", (10, ROI_TOP_PIXELS - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Active zone label
    cv2.putText(frame, "ACTIVE ROAD DETECTION ZONE (LOWER 60%)", (10, ROI_TOP_PIXELS + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if fps > 0:
        cv2.putText(frame, f"Portronics UVC Live @ {fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return frame

def start_web_server(port=8080):
    server = HTTPServer(('0.0.0.0', port), StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server

def run_calibration(src=CAMERA_INDEX, port=8080, force_web=False, snapshot_path=None):
    global latest_jpeg

    # Check for headless environment
    is_headless = force_web or (
        sys.platform.startswith("linux") and 
        "DISPLAY" not in os.environ and 
        "WAYLAND_DISPLAY" not in os.environ
    )

    print("=" * 65)
    print("[CALIBRATION] Starting Portronics USB Webcam Alignment Tool")
    print(f"  Target Device: {src} | Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Horizon Mask: Top {ROI_TOP_PIXELS} px (Upper 40% will be ignored)")
    print("=" * 65)

    # Open camera
    try:
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    except Exception:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"\n[ERROR] Unable to access camera device {src}.")
        print("  - Ensure Portronics USB webcam is plugged into USB port.")
        print("  - Check available video devices with: v4l2-ctl --list-devices")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # If snapshot mode
    if snapshot_path:
        for _ in range(5):  # warm up sensor
            ret, frame = cap.read()
        if ret and frame is not None:
            annotated = annotate_frame(frame)
            cv2.imwrite(snapshot_path, annotated)
            print(f"\n[PASS] Saved calibration snapshot to: {snapshot_path}")
        else:
            print("\n[ERROR] Failed to capture snapshot frame.")
        cap.release()
        return

    # If headless (over SSH) or web requested:
    if is_headless:
        local_ip = get_local_ip()
        start_web_server(port=port)
        print("\n[HEADLESS MODE DETECTED: No desktop display attached]")
        print("🌐 LIVE WEB CALIBRATION SERVER ACTIVE!")
        print(f"   👉 Open this link on your phone/laptop browser:")
        print(f"      http://{local_ip}:{port}")
        print(f"      http://localhost:{port}")
        print("\n   Align camera on windshield until horizon matches the RED line.")
        print("   Press Ctrl+C to terminate.\n")

        prev_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
                prev_time = curr_time

                annotated = annotate_frame(frame, fps=fps)
                _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

                with frame_lock:
                    latest_jpeg = jpeg.tobytes()

                time.sleep(0.03)
        except KeyboardInterrupt:
            print("\n[CALIBRATION] Calibration server stopped.")
        finally:
            cap.release()
            return

    # Desktop GUI mode (when monitor/VNC is connected)
    print("  Instructions: Tilt camera until horizon aligns with the RED line.")
    print("  Press 'q' in the window to exit.")
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
            prev_time = curr_time

            annotated = annotate_frame(frame, fps=fps)

            cv2.imshow("Portronics Webcam Calibration - Press 'q' to Quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"\n[WARN] GUI display error: {e}")
        print("Falling back to web streamer mode...")
        cap.release()
        run_calibration(src=src, port=port, force_web=True)
        return
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[CALIBRATION] Alignment tool closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portronics Webcam Windshield Calibration")
    parser.add_argument("--source", type=int, default=CAMERA_INDEX, help="Camera index (default: 0)")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--web", action="store_true", help="Force web browser streaming mode")
    parser.add_argument("--snapshot", type=str, default=None, help="Save single calibration image to path and exit")
    args = parser.parse_args()

    run_calibration(src=args.source, port=args.port, force_web=args.web, snapshot_path=args.snapshot)
