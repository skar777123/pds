"""
Spatial-Temporal IoU & Centroid Tracker for Road Surface Defects.
Prevents duplicate CSV logging and redundant alerts as the vehicle drives past defects.
"""

import numpy as np

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

class TrackedDefect:
    """Represents a continuous track of a single physical road anomaly."""
    
    def __init__(self, track_id, box, class_id, class_name, severity, confidence):
        self.track_id = track_id
        self.box = box                # [x1, y1, x2, y2]
        self.class_id = class_id
        self.class_name = class_name
        self.severity = severity
        self.confidence = confidence
        self.hits = 1
        self.lost = 0
        self.has_been_logged = False  # Gate to prevent multiple CSV logging

    def update(self, box, class_id, class_name, severity, confidence):
        # Exponential smoothing on bounding box to reduce frame-to-frame jitter
        alpha = 0.6
        self.box = [
            int(alpha * box[i] + (1 - alpha) * self.box[i]) for i in range(4)
        ]
        self.class_id = class_id
        self.class_name = class_name
        self.severity = severity
        self.confidence = max(self.confidence, confidence)
        self.hits += 1
        self.lost = 0

class DefectTracker:
    """Multi-object road defect tracker with spatial-temporal deduplication."""
    
    def __init__(self, iou_threshold=0.3, max_lost=10, min_hits_for_confirm=2):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits_for_confirm = min_hits_for_confirm
        self.next_id = 1
        self.active_tracks = []

    def update(self, detections):
        """
        Update tracker with new frame detections.
        detections: list of dicts: {'box': [x1, y1, x2, y2], 'class_id': int, 
                                   'class_name': str, 'severity': str, 'conf': float}
        Returns:
            confirmed_new_events: list of TrackedDefect that became confirmed on this frame
            current_active_tracks: list of all active TrackedDefect objects
        """
        matched_tracks = set()
        matched_detections = set()
        
        # 1. Match new detections to existing tracks via IoU
        for d_idx, det in enumerate(detections):
            best_iou = 0.0
            best_track_idx = -1
            
            for t_idx, track in enumerate(self.active_tracks):
                if t_idx in matched_tracks:
                    continue
                iou = compute_iou(det['box'], track.box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = t_idx

            if best_iou >= self.iou_threshold and best_track_idx != -1:
                self.active_tracks[best_track_idx].update(
                    det['box'], det['class_id'], det['class_name'], det['severity'], det['conf']
                )
                matched_tracks.add(best_track_idx)
                matched_detections.add(d_idx)

        # 2. Increment lost count for unmatched active tracks
        surviving_tracks = []
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_tracks:
                track.lost += 1
            if track.lost <= self.max_lost:
                surviving_tracks.append(track)
        self.active_tracks = surviving_tracks

        # 3. Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = TrackedDefect(
                    track_id=self.next_id,
                    box=det['box'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    severity=det['severity'],
                    confidence=det['conf']
                )
                self.next_id += 1
                self.active_tracks.append(new_track)

        # 4. Check for newly confirmed physical defects that need logging
        confirmed_new_events = []
        for track in self.active_tracks:
            if track.hits >= self.min_hits_for_confirm and not track.has_been_logged:
                track.has_been_logged = True
                confirmed_new_events.append(track)

        return confirmed_new_events, self.active_tracks
