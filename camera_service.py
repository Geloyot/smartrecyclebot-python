# python/camera_service.py
import cv2
import time
import requests
import os
import sys
from detector import Detector
from dotenv import load_dotenv
from pathlib import Path
from stability_tracker import StabilityTracker
import uuid

# Configuration — tune as needed
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

LARAVEL_WEBHOOK = os.getenv("LARAVEL_WEBHOOK")
DEVICE_API_KEY   = os.getenv("DEVICE_API_KEY")
MODEL_PATH       = os.getenv("MODEL_PATH")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

# Detector & tracker params
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.35))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.6))
CONF_DELTA = float(os.environ.get("CONF_DELTA", 0.05))
REQUIRED_FRAMES = int(os.environ.get("REQUIRED_FRAMES", 5))

def post_event(payload):
    """Send detection event to Laravel"""
    headers = {
        "X-Api-Key": DEVICE_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(
            LARAVEL_WEBHOOK, 
            json=payload, 
            headers=headers, 
            timeout=5
        )
        
        if resp.ok:
            print(f"[POST OK] {payload.get('classification')} score={payload.get('score')}")
            return True
        else:
            print(f"[POST ERR {resp.status_code}] {resp.text}")
            return False
            
    except Exception as e:
        print(f"[POST FAILED] {e}")
        return False

def draw_boxes(frame, detections):
    for d in detections:
        x1,y1,x2,y2 = map(int, (d['xmin'], d['ymin'], d['xmax'], d['ymax']))
        label = f"{d['class_name']}:{d['conf']:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (10,200,10), 2)
        cv2.putText(frame, label, (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

def main():
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
    detector = Detector(model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD)
    tracker = StabilityTracker(iou_threshold=IOU_THRESHOLD, conf_delta=CONF_DELTA, required_frames=REQUIRED_FRAMES)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera. Index:", CAMERA_INDEX)
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed.")
                break

            # Run detection
            detections = detector.detect(frame)

            # Add class_name on detections if missing
            for d in detections:
                if 'class_name' not in d:
                    d['class_name'] = str(d.get('class_id'))

            # Update tracker and check for stable events
            events = tracker.update(detections)
            for ev in events:
                ev['model_name'] = MODEL_PATH
                # Optional: set bin id if your device is mapped to a bin
                # ev['bin_id'] = <your bin id>
                # generate consistent unique request id
                ev['request_id'] = str(uuid.uuid4())
                print("Stable event:", ev)
                post_event(ev)

            # Debug display
            frame_dbg = draw_boxes(frame.copy(), detections)
            cv2.imshow("Detection (press q to quit)", frame_dbg)

            # small sleep to limit CPU (tune as necessary)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
