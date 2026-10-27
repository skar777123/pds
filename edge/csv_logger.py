"""
Atomic Thread-Safe CSV Detection Logger for Pothole Detection System.
Strictly conforms to PRD Section 4 logging requirements.
"""

import os
import csv
import threading
from datetime import datetime, timezone
from edge.config import CSV_LOG_PATH, ENABLE_CONSOLE_LOG

CSV_HEADER = [
    "timestamp_iso8601",
    "track_id",
    "category",
    "severity",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_w",
    "bbox_h",
    "vehicle_speed_approx"
]

class CSVLogger:
    """Handles persistent timestamped defect logging."""
    
    def __init__(self, log_path=CSV_LOG_PATH):
        self.log_path = log_path
        self.lock = threading.Lock()
        self._initialize_csv()

    def _initialize_csv(self):
        """Create log file with header if it doesn't already exist."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
            print(f"[CSVLogger] Created new detection log at: {self.log_path}")

    def log_detection(self, track, approx_speed=40.0):
        """
        Record a newly confirmed defect event.
        track: TrackedDefect instance
        approx_speed: Estimated or GPS vehicle speed in km/h
        """
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        box = track.box  # [x1, y1, x2, y2]
        x1 = box[0]
        y1 = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]

        row = [
            now_iso,
            track.track_id,
            track.class_name,
            track.severity,
            f"{track.confidence:.3f}",
            x1,
            y1,
            w,
            h,
            f"{approx_speed:.1f}"
        ]

        with self.lock:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()

        if ENABLE_CONSOLE_LOG:
            print(f"[EVENT] Time: {now_iso} | ID: #{track.track_id} | Class: {track.class_name} | Severity: {track.severity.upper()} | Conf: {track.confidence:.2f} | Box: [{x1}, {y1}, {w}, {h}]")
