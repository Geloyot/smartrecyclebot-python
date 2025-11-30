# python/detector.py
import gc
import numpy as np
from PIL import Image
import io

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics package not available. Install with `pip install ultralytics`") from e

class Detector:
    def __init__(self, model_path: str = "models/best.onnx", conf_threshold: float = 0.30):
        """
        model_path: path to your YOLO weights (or 'best.onnx')
        conf_threshold: minimum confidence to consider
        """
        self.model = YOLO(model_path, task='detect')
        self.conf_threshold = conf_threshold
        self.names = self.model.names if hasattr(self.model, "names") else {}
        
        print(f"✓ Detector initialized: {model_path}")
        print(f"✓ Classes loaded: {len(self.names)}")
        
        # Force garbage collection after model loading
        gc.collect()

    def detect(self, frame=None, image_bytes=None):
        """
        Run one inference on the provided frame or image bytes.
        
        Args:
            frame: BGR numpy array from OpenCV (optional)
            image_bytes: Raw image bytes (optional)
        
        Returns:
            List of detections: each is dict {xmin,ymin,xmax,ymax,conf,class_id,class_name}
        """
        try:
            # Handle different input types
            if image_bytes is not None:
                # Convert bytes to numpy array
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                frame = np.array(pil_image)
                # Convert RGB to BGR for YOLO (if needed, though ONNX might handle RGB)
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if frame is None:
                raise ValueError("Either frame or image_bytes must be provided")
            
            # Run inference with memory-optimized settings
            results = self.model.predict(
                frame, 
                imgsz=640,  # Reduced from 640 to save memory (adjust based on your needs)
                conf=self.conf_threshold,
                verbose=False,
                device='cpu'  # Explicitly use CPU
            )
            
            # Extract detections
            detections = []
            res0 = results[0]
            
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
                        "bbox": [xmin, ymin, xmax, ymax]  # Added for compatibility with app.py
                    })
            
            # Clean up to free memory
            del results
            del res0
            gc.collect()
            
            return detections
            
        except Exception as e:
            # Clean up even on error
            gc.collect()
            raise e