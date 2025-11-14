# python/detector.py
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics package not available. Install with `pip install ultralytics`") from e

class Detector:
    def __init__(self, model_path: str = "models/best.pt", conf_threshold: float = 0.30):
        """
        model_path: path to your YOLO weights (or 'best.pt')
        conf_threshold: minimum confidence to consider
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame: np.ndarray):
        """
        Run one inference on the provided frame (BGR numpy array from OpenCV).
        Returns list of detections: each is dict {xmin,ymin,xmax,ymax,conf,class_id,class_name}
        """
        # Ultralytics accepts numpy arrays, pass frame (BGR) directly
        results = self.model.predict(frame, imgsz=640, conf=self.conf_threshold, verbose=False)
        # results is list-like; take first
        res0 = results[0]
        detections = []
        # res0.boxes.xyxy, res0.boxes.conf, res0.boxes.cls
        if hasattr(res0, "boxes") and len(res0.boxes) > 0:
            boxes = res0.boxes.xyxy.cpu().numpy()  # Nx4
            confs = res0.boxes.conf.cpu().numpy()  # N
            classes = res0.boxes.cls.cpu().numpy().astype(int)  # N
            for (xyxy, conf, cls) in zip(boxes, confs, classes):
                xmin, ymin, xmax, ymax = [float(x) for x in xyxy]
                detections.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "conf": float(conf),
                    "class_id": int(cls),
                    "class_name": self.names.get(int(cls), str(int(cls))),
                })
        return detections
