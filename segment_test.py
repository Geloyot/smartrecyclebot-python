#!/usr/bin/env python3
"""
segment_test_fixed_display.py

- Runs YOLOv8-seg on a square padded input (640x640) to avoid bogus full-image masks.
- Maps masks back to original camera resolution (no showing padded square).
- Draws semi-transparent overlays on the original frame.
- Only 'q' quits the program.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# CPU optimizations (non-exhaustive)
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = False
torch.backends.mkldnn.enabled = True

# -------- CONFIG ----------
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 640     # camera capture resolution (use your chosen)
MODEL_PATH = "models/best_segment.pt"
INP_SIZE = 640                  # square input expected by segmentation model
CONF = 0.25
IOU = 0.45
# skip masks that still (after cropping) cover too much of the resized area (likely bogus)
MASK_AREA_SKIP_RATIO = 0.95
# --------------------------

palette = [tuple(map(int, c)) for c in np.random.randint(30, 230, size=(64, 3))]

def preprocess_to_square(frame, size=INP_SIZE):
    """Resize frame preserving aspect ratio, then pad to square (bottom-right padding)."""
    h, w = frame.shape[:2]
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_right = size - new_w
    pad_bottom = size - new_h
    # pad order: top, bottom, left, right
    padded = cv2.copyMakeBorder(resized, 0, pad_bottom, 0, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    meta = {
        "orig_w": w, "orig_h": h,
        "scale": scale,
        "resized_w": new_w, "resized_h": new_h,
        "pad_right": pad_right, "pad_bottom": pad_bottom
    }
    return padded, meta

def unpad_and_upscale_mask(mask_square, meta):
    """
    mask_square: HxW mask at INP_SIZE x INP_SIZE (0/1 or 0/255)
    Returns mask scaled to original frame size (orig_w x orig_h).
    We remove the bottom-right padding first, then upsample to original resolution.
    """
    # ensure binary 0/255
    m = (mask_square > 0).astype(np.uint8) * 255
    # crop region corresponding to resized image (no padding)
    rh, rw = meta["resized_h"], meta["resized_w"]
    cropped = m[:rh, :rw]
    # upsample (nearest to preserve binary)
    full = cv2.resize(cropped, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_NEAREST)
    return full

def overlay_mask(img, mask, color, alpha=0.35):
    """In-place semi-transparent mask overlay (mask should be 0/255 uint8 same size as img)."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return img
    colored = np.zeros_like(img, dtype=np.uint8)
    colored[:] = color
    blended = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
    img[mask_bool] = blended[mask_bool]
    return img

def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = int(xs.mean())
    cy = int(ys.mean())
    return (cx, cy)

def safe_to_numpy(mask_obj):
    """Convert model mask object to numpy HxW (supports torch tensor or numpy)."""
    try:
        # common: torch tensor (H,W) or (N,H,W). If it is a tensor, convert.
        if hasattr(mask_obj, "cpu"):
            arr = mask_obj.cpu().numpy()
        else:
            arr = np.array(mask_obj)
    except Exception:
        arr = np.array(mask_obj)
    # ensure 2D
    if arr.ndim == 2:
        return arr
    # if arr is float probabilities, binarize at 0.5
    if arr.ndim == 3 and arr.shape[0] == 1:
        return (arr[0] > 0.5).astype(np.uint8) * 255
    # fallback: if shape not expected, try squeeze
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return (arr > 0.5).astype(np.uint8) * 255
    # unknown -> empty
    return np.zeros((INP_SIZE, INP_SIZE), dtype=np.uint8)

def main():
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    model.to("cpu")

    # open camera
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # wait for camera warmup
    warmed = False
    warm_start = time.time()
    while time.time() - warm_start < 5.0:
        ret, frame = cap.read()
        if ret and frame is not None:
            warmed = True
            break
    if not warmed:
        print("ERROR: camera failed to produce frames. Check camera index and permissions.")
        cap.release()
        return

    print("Camera ready. Press 'q' to quit.")
    window = "Segment Test"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame available.")
            time.sleep(0.02)
            continue

        # ---- preprocess to square ----
        inp, meta = preprocess_to_square(frame, size=INP_SIZE)

        # run model on square input
        # imgsz=INP_SIZE avoids stride warnings
        results = model(inp, device="cpu", imgsz=INP_SIZE, conf=CONF, iou=IOU, verbose=False)
        r = results[0]

        disp = frame.copy()

        # masks handling
        masks_obj = getattr(r, "masks", None)
        if masks_obj is None:
            # nothing detected: still show frame
            cv2.imshow(window, disp)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
            continue

        # Ultralytics returns masks in r.masks.data typically shape (N, INP_SIZE, INP_SIZE)
        try:
            masks_data = masks_obj.data
        except Exception:
            masks_data = masks_obj

        # force iterable list
        masks_list = []
        # if it's a single mask -> make it list
        if hasattr(masks_data, "ndim") and masks_data.ndim == 2:
            masks_list = [masks_data]
        else:
            try:
                # iterate over first dimension
                for i in range(masks_data.shape[0]):
                    masks_list.append(masks_data[i])
            except Exception:
                # fallback: try direct iterable
                try:
                    for m in masks_data:
                        masks_list.append(m)
                except Exception:
                    masks_list = []

        # classes/conf (defensive)
        cls_list = []
        conf_list = []
        try:
            if hasattr(r, "boxes") and r.boxes is not None:
                cls_arr = r.boxes.cls
                conf_arr = r.boxes.conf
                # convert to numpy if tensors
                if hasattr(cls_arr, "cpu"):
                    cls_list = cls_arr.cpu().numpy().astype(int).tolist()
                else:
                    cls_list = np.array(cls_arr).astype(int).tolist()
                if hasattr(conf_arr, "cpu"):
                    conf_list = conf_arr.cpu().numpy().tolist()
                else:
                    conf_list = np.array(conf_arr).tolist()
        except Exception:
            cls_list = []
            conf_list = []

        # process each mask: unpad -> upscale -> overlay -> label at centroid
        for i, m in enumerate(masks_list):
            mask_sq = safe_to_numpy(m)  # INP_SIZE x INP_SIZE
            # check area on the *resized (unpadded) area* to avoid skipping legitimate large objects:
            # crop to resized dims
            resized_h, resized_w = meta["resized_h"], meta["resized_w"]
            if mask_sq.shape[0] != INP_SIZE or mask_sq.shape[1] != INP_SIZE:
                # ensure shape matches expected; resize nearest
                mask_sq = cv2.resize(mask_sq.astype(np.uint8), (INP_SIZE, INP_SIZE), interpolation=cv2.INTER_NEAREST)
            cropped = mask_sq[:resized_h, :resized_w]
            area_ratio = (cropped > 0).sum() / float(cropped.size)
            if area_ratio > MASK_AREA_SKIP_RATIO:
                # skip suspiciously large masks (usually model error)
                # (we don't print every frame to avoid flooding the console)
                continue

            # unpad and upscale to original frame size
            mask_full = unpad_and_upscale_mask(mask_sq, meta)  # orig_h x orig_w

            # overlay
            color = palette[i % len(palette)]
            overlay_mask(disp, mask_full, color, alpha=0.35)

            # contour
            try:
                contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(disp, contours, -1, color, 2)
            except Exception:
                pass

            # centroid + label text
            centroid = mask_centroid(mask_full)
            cls = cls_list[i] if i < len(cls_list) else None
            conf = conf_list[i] if i < len(conf_list) else None
            label = ""
            if cls is not None:
                label = f"{cls}"
            if conf is not None:
                try:
                    label += f" {conf:.2f}"
                except Exception:
                    label += f" {conf}"
            if centroid is not None:
                cx, cy = centroid
                cv2.circle(disp, (cx, cy), 6, (255,255,255), -1)
                if label:
                    cv2.putText(disp, label, (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow(window, disp)
        # only quit on q or Q
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
