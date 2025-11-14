# python/stability_tracker.py
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math

def iou(boxA, boxB):
    # boxes: dicts with xmin,ymin,xmax,ymax
    xA = max(boxA['xmin'], boxB['xmin'])
    yA = max(boxA['ymin'], boxB['ymin'])
    xB = min(boxA['xmax'], boxB['xmax'])
    yB = min(boxA['ymax'], boxB['ymax'])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA['xmax'] - boxA['xmin']) * (boxA['ymax'] - boxA['ymin'])
    boxBArea = (boxB['xmax'] - boxB['xmin']) * (boxB['ymax'] - boxB['ymin'])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

@dataclass
class Tracked:
    id: str
    bbox: Dict
    cls: int
    conf_avg: float
    stable_count: int = 0
    last_seen: float = field(default_factory=time.time)
    sent: bool = False  # whether we already sent classification for this tracked object

class StabilityTracker:
    def __init__(self, iou_threshold=0.6, conf_delta=0.05, required_frames=5, max_unseen_seconds=1.0):
        """
        iou_threshold: minimum IoU between matched boxes to count as same object
        conf_delta: maximum allowed absolute difference in confidence to still be considered stable
        required_frames: number of consecutive frames meeting criteria to send
        max_unseen_seconds: how long to keep tracked objects without matches
        """
        self.tracked: List[Tracked] = []
        self.iou_threshold = iou_threshold
        self.conf_delta = conf_delta
        self.required_frames = required_frames
        self.max_unseen_seconds = max_unseen_seconds

    def update(self, detections: List[Dict]):
        """
        Accepts new detections (list of dicts with bbox,conf,class_name,class_id).
        Returns list of 'stable events' i.e. payload dicts to be sent to server.
        """
        now = time.time()
        stable_events = []

        # Mark all tracked as not updated this frame
        updated_indices = set()

        # For each detection, try to match to existing tracked by highest IoU
        for det in detections:
            best_idx = None
            best_iou = 0.0
            for idx, t in enumerate(self.tracked):
                if t.cls != det['class_id']:
                    continue
                val = iou(t.bbox, det)
                if val > best_iou:
                    best_iou = val
                    best_idx = idx
            if best_idx is not None and best_iou >= self.iou_threshold:
                t = self.tracked[best_idx]
                # update average confidence (simple running average)
                t.conf_avg = (t.conf_avg + det['conf']) / 2.0
                # update bbox to new bbox (you could smooth if desired)
                t.bbox = det
                t.last_seen = now
                # check confidence difference
                if abs(t.conf_avg - det['conf']) <= self.conf_delta:
                    t.stable_count += 1
                else:
                    t.stable_count = 0
                updated_indices.add(best_idx)
            else:
                # not matched -> create new tracked object
                new_t = Tracked(
                    id=str(uuid.uuid4()),
                    bbox=det,
                    cls=det['class_id'],
                    conf_avg=det['conf'],
                    stable_count=1,
                    last_seen=now,
                    sent=False
                )
                self.tracked.append(new_t)

        # For tracked objects not matched this frame, reset stable_count or check expiry
        for idx, t in enumerate(self.tracked):
            if idx not in updated_indices:
                # If not matched recently, check expiry
                if now - t.last_seen > self.max_unseen_seconds:
                    # mark for removal later
                    t.stable_count = 0
                # else: keep tracked, but don't increment stable count

        # Collect events from those tracked that meet required_frames and haven't been sent
        for t in list(self.tracked):
            if t.stable_count >= self.required_frames and not t.sent:
                # Build payload — include class_name & score (avg)
                payload = {
                    "request_id": str(uuid.uuid4()),
                    "bin_id": None,  # if you want to map device to bin, fill here
                    "classification": t.bbox.get("class_name", str(t.cls)),
                    "score": float(t.conf_avg),
                    "model_name": None,  # set by caller (camera_service)
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t.last_seen)),
                    "tracked_id": t.id,
                }
                stable_events.append(payload)
                t.sent = True  # avoid repeated events for same object
            # Remove stale tracked objects that haven't been seen recently
            if now - t.last_seen > (self.max_unseen_seconds * 5):
                self.tracked.remove(t)

        return stable_events
