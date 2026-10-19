"""
Asynchronous Camera Capture Worker for Portronics USB Webcam (V4L2 / UVC).
Prevents USB buffer lag on Raspberry Pi by reading frames continuously in a background thread.
"""

import time
import threading
import cv2
from edge.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, FOURCC

class CameraStream:
    """Threaded camera capture to ensure zero frame-buffering lag on embedded Linux."""
    
    def __init__(self, src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=TARGET_FPS):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Initialize camera capture and start background polling thread."""
        print(f"[CameraStream] Initializing video source {self.src} at {self.width}x{self.height} @ {self.fps} FPS...")
        
        # Use V4L2 on Linux if available, or default
        try:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        except Exception:
            self.cap = cv2.VideoCapture(self.src)
            
        if not self.cap.isOpened():
            # Fallback to standard index
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            raise RuntimeError(f"[CameraStream] ERROR: Unable to open webcam at index {self.src}. Check USB connection.")

        # Set hardware parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        except Exception:
            pass

        # Warm up sensor
        self.ret, self.frame = self.cap.read()
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print("[CameraStream] Camera capture thread active.")
        return self

    def _update(self):
        """Continuously grab newest frame, discarding stale frames."""
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def read(self):
        """Return the latest frame immediately without waiting for hardware sync."""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        """Release camera resources and terminate background worker."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        print("[CameraStream] Camera stream terminated.")
