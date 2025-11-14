# app.py

import os
import io
import time
from typing import Any, Dict, List, Optional

from camera_controller import start_camera_thread, stop_camera_thread, camera_status
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# image helpers
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load .env
load_dotenv()

# import your existing modules
from detector import Detector

# --- Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
SERVICE_PORT = int(os.getenv("PYTHON_SERVICE_PORT", 8000))

app = FastAPI(title="SmartRecyclebot Python Service")

# Add CORS middleware for Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Laravel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# instantiate detector
try:
    detector = Detector(model_path=MODEL_PATH)
    logger.info(f"Detector initialized with model: {MODEL_PATH}")
except TypeError:
    # fallback: Detector() without args
    detector = Detector()
    logger.warning("Detector initialized without model path")
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    detector = None

# in-memory last result store
_latest_result: Optional[Dict[str, Any]] = None


# Pydantic response models
class PredictResponse(BaseModel):
    timestamp: float
    detections: List[Dict[str, Any]]


class CameraStatusResponse(BaseModel):
    running: bool
    last_result: Optional[Dict[str, Any]]


class MessageResponse(BaseModel):
    message: str
    status: Optional[str] = None


def _call_detect_with_fallbacks(image_bytes: bytes):
    """
    Try common ways to call detector.detect:
      1) detector.detect(image_bytes=...)
      2) detector.detect(image_bytes)
      3) detector.detect(pil_image)
      4) detector.detect(np_array)
    Return whatever detector returns (best-effort).
    """
    if detector is None:
        raise RuntimeError("Detector not initialized")
    
    # 1) raw bytes named param
    try:
        return detector.detect(image_bytes=image_bytes)
    except Exception:
        pass

    # 2) raw bytes positional
    try:
        return detector.detect(image_bytes)
    except Exception:
        pass

    # convert to PIL
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to decode image bytes: {e}")

    # 3) pass PIL image
    try:
        return detector.detect(pil)
    except Exception:
        pass

    # 4) pass numpy array (HxWxC)
    try:
        np_img = np.array(pil)
        return detector.detect(np_img)
    except Exception:
        pass

    raise RuntimeError("Could not call detector.detect with any supported signature.")


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "SmartRecycleBot Camera Service",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "camera_start": "/camera/start",
            "camera_stop": "/camera/stop",
            "camera_status": "/camera/status",
            "camera_latest": "/camera/latest",
            "infer": "/infer"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    cam_status = camera_status()
    return {
        "status": "ok",
        "detector_loaded": detector is not None,
        "camera_running": cam_status.get("running", False)
    }


@app.post("/infer", response_model=PredictResponse)
async def infer(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (multipart/form-data).
    Runs detection and returns detections.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        detections = _call_detect_with_fallbacks(contents)
        logger.info(f"Detection completed: {len(detections) if isinstance(detections, list) else 'N/A'} objects")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    # normalize result to a dict with 'detections' key if possible
    if isinstance(detections, dict):
        result = detections
    else:
        # if detector returns a list, wrap it
        result = {"detections": detections}

    # attach a timestamp
    result_with_meta = {
        "timestamp": time.time(),
        "detections": result.get("detections", result)
    }

    # update in-memory cache
    global _latest_result
    _latest_result = result_with_meta

    return result_with_meta


@app.get("/predict_latest", response_model=PredictResponse)
def predict_latest():
    """Return the last prediction performed via /infer (in-memory)."""
    if _latest_result is None:
        raise HTTPException(status_code=404, detail="No predictions yet")
    return _latest_result


@app.post("/camera/start", response_model=MessageResponse)
def camera_start():
    """Start the camera detection thread"""
    logger.info("[API] Camera start requested")
    res = start_camera_thread()
    
    if res.get("started"):
        logger.info("[API] Camera started successfully")
        return MessageResponse(
            message=res.get("message", "Camera started"),
            status="started"
        )
    
    # Already running or error
    status_code = 200 if "already" in res.get("message", "").lower() else 409
    logger.warning(f"[API] Camera start failed: {res.get('message')}")
    raise HTTPException(status_code=status_code, detail=res.get("message"))


@app.post("/camera/stop", response_model=MessageResponse)
def camera_stop():
    """Stop the camera detection thread"""
    logger.info("[API] Camera stop requested")
    res = stop_camera_thread()
    
    if res.get("stopped"):
        logger.info("[API] Camera stopped successfully")
        return MessageResponse(
            message=res.get("message", "Camera stopped"),
            status="stopped"
        )
    
    logger.warning(f"[API] Camera stop failed: {res.get('message')}")
    raise HTTPException(status_code=400, detail=res.get("message"))


@app.get("/camera/status", response_model=CameraStatusResponse)
def camera_status_route():
    """Get current camera status and last result"""
    status = camera_status()
    return CameraStatusResponse(
        running=status.get("running", False),
        last_result=status.get("last_result")
    )


@app.get("/camera/latest")
def camera_latest():
    """Get only the latest camera detection result"""
    status = camera_status()
    if not status.get("last_result"):
        raise HTTPException(status_code=404, detail="No results yet")
    return status["last_result"]


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("SMART RECYCLEBOT - Python Camera Service")
    logger.info("=" * 60)
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Service Port: {SERVICE_PORT}")
    logger.info(f"Detector Status: {'Loaded' if detector else 'Not Loaded'}")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[SHUTDOWN] Stopping camera service...")
    cam_status = camera_status()
    if cam_status.get("running"):
        stop_camera_thread()
        logger.info("[SHUTDOWN] Camera stopped")
    logger.info("[SHUTDOWN] Service stopped")


# If run as script
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on http://0.0.0.0:{SERVICE_PORT}")
    uvicorn.run("app:app", host="0.0.0.0", port=SERVICE_PORT, reload=True)