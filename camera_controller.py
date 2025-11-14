# camera_controller.py
import threading
import time
import cv2
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from detector import Detector
from stability_tracker import StabilityTracker
import camera_service

load_dotenv()

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.6))
CONF_DELTA = float(os.getenv("CONF_DELTA", 0.05))
REQUIRED_FRAMES = int(os.getenv("REQUIRED_FRAMES", 5))

_camera_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_last_result: Optional[Dict[str, Any]] = None
_detector: Optional[Detector] = None
_tracker: Optional[StabilityTracker] = None
_camera_cap: Optional[cv2.VideoCapture] = None
_lock = threading.Lock()

def get_detector() -> Detector:
    global _detector
    if _detector is None:
        conf_threshold = float(os.getenv("CONF_THRESHOLD", 0.35))
        try:
            _detector = Detector(model_path=MODEL_PATH, conf_threshold=conf_threshold)
            print(f"[Detector] Initialized with model: {MODEL_PATH}")
        except Exception as e:
            print(f"[Detector] Initialization error: {e}")
            raise
    return _detector

def get_tracker() -> StabilityTracker:
    global _tracker
    if _tracker is None:
        _tracker = StabilityTracker(
            iou_threshold=IOU_THRESHOLD,
            conf_delta=CONF_DELTA,
            required_frames=REQUIRED_FRAMES,
            max_unseen_seconds=1.0
        )
        print(f"[Tracker] Initialized (IOU: {IOU_THRESHOLD}, Frames: {REQUIRED_FRAMES})")
    return _tracker

def open_camera():
    global _camera_cap
    if _camera_cap is not None:
        return _camera_cap
    
    print(f"[Camera] Opening camera at index {CAMERA_INDEX}")
    _camera_cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not _camera_cap.isOpened():
        print(f"[Camera] Failed to open camera at index {CAMERA_INDEX}")
        _camera_cap = None
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
    
    print("[Camera] Camera opened successfully")
    return _camera_cap

def close_camera():
    global _camera_cap
    if _camera_cap is not None:
        _camera_cap.release()
        _camera_cap = None
        print("[Camera] Camera closed")

def reset_tracker():
    """Reset the tracker when camera stops"""
    global _tracker
    _tracker = None

def camera_loop(stop_event: threading.Event, poll_interval: float = 0.033):
    """
    Camera loop with detection, stability tracking, and webhook posting.
    Only sends stable detections (objects seen consistently for REQUIRED_FRAMES).
    """
    print("[camera_loop] Starting camera loop")
    
    try:
        det = get_detector()
        tracker = get_tracker()
        cap = open_camera()
    except Exception as e:
        print(f"[camera_loop] Failed to initialize: {e}")
        return
    
    frame_count = 0
    total_detections_sent = 0
    
    try:
        while not stop_event.is_set():
            try:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    print("[camera_loop] Failed to read frame")
                    time.sleep(poll_interval)
                    continue
                
                frame_count += 1
                
                # Run detection
                detections = det.detect(frame)
                
                # Ensure class_name exists in each detection
                for d in detections:
                    if 'class_name' not in d:
                        d['class_name'] = str(d.get('class_id'))
                
                # Pass detections through stability tracker
                # This returns only STABLE events (objects seen consistently)
                stable_events = tracker.update(detections)
                
                # Create result object for status endpoint
                result = {
                    "timestamp": time.time(),
                    "detections": detections,
                    "stable_count": len(stable_events),
                    "frame_number": frame_count,
                    "total_sent": total_detections_sent
                }
                
                # Save last result thread-safely
                with _lock:
                    global _last_result
                    _last_result = result
                
                # Post only STABLE events to Laravel webhook
                if stable_events and hasattr(camera_service, 'post_event'):
                    for event in stable_events:
                        # Add model_name (tracker sets it to None)
                        event['model_name'] = MODEL_PATH
                        
                        try:
                            success = camera_service.post_event(event)
                            if success:
                                total_detections_sent += 1
                                print(f"[camera_loop] ✓ STABLE: {event['classification']} "
                                      f"({event['score']:.2f}) "
                                      f"[tracked_id: {event.get('tracked_id', 'N/A')[:8]}...]")
                            else:
                                print(f"[camera_loop] ✗ Failed to post: {event['classification']}")
                        except Exception as e:
                            print(f"[camera_loop] post_event error: {e}")
                
                # Optional: Log frame info every 30 frames (~1 second at 30 FPS)
                if frame_count % 30 == 0:
                    print(f"[camera_loop] Frame {frame_count}: "
                          f"{len(detections)} detections, "
                          f"{len(stable_events)} stable, "
                          f"{total_detections_sent} total sent")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"[camera_loop] Frame processing error: {e}")
                time.sleep(poll_interval)
                
    except Exception as e:
        print(f"[camera_loop] Fatal error: {e}")
    finally:
        close_camera()
        reset_tracker()
        print(f"[camera_loop] Exiting (sent {total_detections_sent} stable detections)")

def start_camera_thread(poll_interval: float = 0.033) -> Dict[str, Any]:
    """Start the camera detection thread"""
    global _camera_thread, _stop_event
    
    with _lock:
        if _camera_thread and _camera_thread.is_alive():
            return {"started": False, "message": "Camera already running"}
        
        print("[start_camera_thread] Starting camera thread")
        reset_tracker()  # Ensure clean tracker state
        _stop_event = threading.Event()
        _camera_thread = threading.Thread(
            target=camera_loop, 
            args=(_stop_event, poll_interval), 
            daemon=True
        )
        _camera_thread.start()
        
        return {"started": True, "message": "Camera thread started"}

def stop_camera_thread(timeout: float = 5.0) -> Dict[str, Any]:
    """Stop the camera detection thread"""
    global _camera_thread, _stop_event
    
    with _lock:
        if not _camera_thread or not _camera_thread.is_alive():
            return {"stopped": False, "message": "Camera not running"}
        
        print("[stop_camera_thread] Stopping camera thread")
        _stop_event.set()
    
    _camera_thread.join(timeout)
    
    if _camera_thread.is_alive():
        print("[stop_camera_thread] Failed to stop within timeout")
        return {"stopped": False, "message": "Failed to stop within timeout"}
    else:
        reset_tracker()  # Clean up tracker
        print("[stop_camera_thread] Camera stopped successfully")
        return {"stopped": True, "message": "Camera stopped"}

def camera_status() -> Dict[str, Any]:
    """Get current camera status and last result"""
    with _lock:
        running = bool(_camera_thread and _camera_thread.is_alive())
        last = _last_result
    
    return {
        "running": running,
        "last_result": last
    }
