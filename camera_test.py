# pip install opencv-python ultralytics
import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import xml.etree.ElementTree as ET

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best.pt"
OUTPUT_DIR = "images"
CAM_INDEX = 0
IMGZ = 640  # inference size

BIODEGRADABLE_CLASSES = [0]
NON_BIODEGRADABLE_CLASSES = [1]
CONF_BIODEGRADABLE = 0.25
CONF_NORMAL = 0.25

# Visual options
USE_MANUAL_DRAW = True
USE_BLEND_FALLBACK = True
BLEND_ALPHA = 0.75  # when blending annotated plot with original

# Save annotation format: "yolo", "xml", or "both"
ANNOTATION_FORMAT = "xml"

# Colors (BGR)
CLASS_COLORS = {
    0: (0, 200, 0),    # biodegradable -> green
    1: (128, 0, 0),    # non-biodegradable -> navy-blue-ish
}
TEXT_COLOR = (255, 255, 255)
TEXT_BG_ALPHA = 0.6
LEGEND_PADDING = 4

# Option to save screenshot with bounding boxes
SAVE_WITH_BOUNDING_BOXES_DEFAULT = False # Default state
# -----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- helpers ----------
def safe_to_numpy(x):
    try:
        return x.cpu().numpy()
    except Exception:
        return np.asarray(x)


def draw_legend(img, names, class_colors):
    out = img.copy()
    if names is None:
        return out
    x = LEGEND_PADDING
    y = LEGEND_PADDING
    line_h = 24
    for cid, color in class_colors.items():
        try:
            cls_name = names[cid] if isinstance(names, (list, tuple)) else names.get(cid, str(cid))
        except Exception:
            cls_name = str(cid)
        cv2.rectangle(out, (x, y), (x + 18, y + 18), color, -1)
        cv2.putText(out, cls_name, (x + 26, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
        y += line_h
    return out


def draw_detections_on_frame(frame, results, names=None):
    out = frame.copy()
    if not results or len(results) == 0:
        return out
    r = results[0]
    if not hasattr(r, "boxes") or len(r.boxes) == 0:
        return out

    xy = getattr(r.boxes, "xyxy", None)
    confs = getattr(r.boxes, "conf", None)
    cls = getattr(r.boxes, "cls", None)
    if xy is None or confs is None or cls is None:
        return out

    xy = safe_to_numpy(xy)
    confs = safe_to_numpy(confs).ravel()
    cls = safe_to_numpy(cls).ravel().astype(int)

    h, w = out.shape[:2]
    for (x1, y1, x2, y2), conf, cid in zip(xy, confs, cls):
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        color = CLASS_COLORS.get(int(cid), (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # label
        label_name = None
        if names is not None:
            try:
                label_name = names[int(cid)]
            except Exception:
                try:
                    label_name = names.get(int(cid), str(int(cid)))
                except Exception:
                    label_name = str(int(cid))
        label_text = f"{label_name if label_name is not None else cid} {conf:.2f}"

        ((tw, th), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx, ty = x1, max(12, y1 - 6)
        bg_x1, bg_y1 = tx, ty - th - 4
        bg_x2, bg_y2 = tx + tw + 6, ty + 2
        bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
        bg_x2, bg_y2 = min(w - 1, bg_x2), min(h - 1, bg_y2)

        overlay = out.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, TEXT_BG_ALPHA, out, 1 - TEXT_BG_ALPHA, 0, out)
        cv2.putText(out, label_text, (bg_x1 + 3, bg_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

    return out


def collect_boxes(results_list):
    """
    Merge boxes from multiple ultralytics results lists.
    Returns list of dicts: [{'class': id, 'conf': f, 'xyxy': (x1,y1,x2,y2)}, ...]
    """
    boxes = []
    for results in results_list:
        if not results or len(results) == 0:
            continue
        r = results[0]
        if not hasattr(r, "boxes") or len(r.boxes) == 0:
            continue
        xy = getattr(r.boxes, "xyxy", None)
        confs = getattr(r.boxes, "conf", None)
        cls = getattr(r.boxes, "cls", None)
        if xy is None or confs is None or cls is None:
            continue
        xy = safe_to_numpy(xy)
        confs = safe_to_numpy(confs).ravel()
        cls = safe_to_numpy(cls).ravel().astype(int)
        for (x1, y1, x2, y2), conf, cid in zip(xy, confs, cls):
            boxes.append({
                "class": int(cid),
                "conf": float(conf),
                "xyxy": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            })
    return boxes


def save_yolo_annotation(path_base, boxes, image_shape):
    """
    Save YOLO-format .txt file next to image.
    path_base: full path without extension, e.g. /.../screenshot_...
    boxes: list of dicts with 'class' and 'xyxy'
    image_shape: (h, w, c)
    """
    h, w = image_shape[:2]
    txt_path = path_base + ".txt"
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{b['class']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    # always write file (may be empty) to indicate screenshot processed
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    return txt_path


def save_voc_xml(path_base, boxes, image_filename, image_shape, folder_name=None):
    """
    Save Pascal VOC-style XML annotation next to image.
    path_base: full path without extension
    boxes: list of dicts with 'class' and 'xyxy'
    image_filename: basename with extension
    image_shape: (h,w,c)
    """
    h, w = image_shape[:2]
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder_name or os.path.basename(os.path.dirname(path_base))
    ET.SubElement(root, "filename").text = image_filename
    ET.SubElement(root, "path").text = path_base + os.path.splitext(image_filename)[1]
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(image_shape[2] if len(image_shape) > 2 else 1)

    ET.SubElement(root, "segmented").text = "0"

    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        obj = ET.SubElement(root, "object")
        # name from model.names will be filled in caller (pass in names if needed)
        ET.SubElement(obj, "name").text = str(b["class"])
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)

    tree = ET.ElementTree(root)
    xml_path = path_base + ".xml"
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


# ---------- model + camera init ----------
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded. Names:", model.names)
except Exception as e:
    print("❌ Failed to load model:", e)
    raise SystemExit(1)

cam = cv2.VideoCapture(CAM_INDEX)
if not cam.isOpened():
    print("❌ Could not open camera index", CAM_INDEX)
    raise SystemExit(1)

TARGET_W, TARGET_H = 1280, 720

# Request 1080p capture from the camera
cam.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

# Read back actual capture size (driver may ignore request)
actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
print(f"Camera capture resolution: {actual_w}x{actual_h} (requested {TARGET_W}x{TARGET_H})")

# Create a resizable display window and size it to the actual capture resolution
cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
# If the camera honored the 1080p request use target, otherwise use actual reported size
win_w, win_h = (TARGET_W, TARGET_H) if (actual_w == TARGET_W and actual_h == TARGET_H) else (max(640, actual_w), max(360, actual_h))
cv2.resizeWindow("Live Camera", win_w, win_h)

print("\n--- Live Camera ---  Press 'q' to quit, 's' to save screenshot (and annotation).\n")

# ---------- main loop ----------
try:
    save_with_boxes = SAVE_WITH_BOUNDING_BOXES_DEFAULT

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        # keep a clean copy of the raw frame BEFORE any drawing/annotation
        orig_frame = frame.copy()

        # --- run detections (same as before) ---
        res_bio = model.track(frame, classes=BIODEGRADABLE_CLASSES, conf=CONF_BIODEGRADABLE, persist=True, imgsz=IMGZ)
        res_non = model.track(frame, classes=NON_BIODEGRADABLE_CLASSES, conf=CONF_NORMAL, persist=True, imgsz=IMGZ)

        # compute totals
        total = 0
        try:
            total = (len(res_bio[0].boxes) if res_bio and hasattr(res_bio[0], "boxes") else 0) + \
                    (len(res_non[0].boxes) if res_non and hasattr(res_non[0], "boxes") else 0)
        except Exception:
            total = 0

        # build display (manual draw preferred)
        display = None
        if USE_MANUAL_DRAW:
            try:
                display = draw_detections_on_frame(frame, res_bio, names=model.names)
                display = draw_detections_on_frame(display, res_non, names=model.names)
            except Exception as e:
                print("⚠️ Manual draw failed (falling back):", e)
                USE_MANUAL_DRAW = False
                display = None

        if display is None:
            try:
                ann = res_bio[0].plot()
                ann = res_non[0].plot(img=ann)
                if USE_BLEND_FALLBACK:
                    display = cv2.addWeighted(frame, BLEND_ALPHA, ann, 1.0 - BLEND_ALPHA, 0)
                else:
                    display = ann
            except Exception as e:
                print("❌ plot() fallback failed:", e)
                display = frame.copy()

        # overlay legend and total
        display = draw_legend(display, model.names, CLASS_COLORS)

        # Show current screenshot mode and total
        mode_text = "BOXES ON" if save_with_boxes else "BOXES OFF"
        cv2.putText(display, f"Total: {total}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"Save Mode: {mode_text}", (50, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Live Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # New: Toggle save mode
        if key == ord("b"):
            save_with_boxes = not save_with_boxes
            print(f"🔄 Screenshot mode toggled. Now saving with boxes: {'ON' if save_with_boxes else 'OFF'}.")
            continue # Skip saving logic for a toggle

        if key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base = os.path.join(OUTPUT_DIR, f"screenshot_{timestamp}")

            # 1) Select the frame to save based on the toggle state
            frame_to_save = display if save_with_boxes else orig_frame
            save_mode_desc = "boxes" if save_with_boxes else "raw"

            img_path = base + ".png"
            cv2.imwrite(img_path, frame_to_save)
            print(f"📸 Saved {save_mode_desc} image (boxes={'ON' if save_with_boxes else 'OFF'}):", img_path)

            # 2) Collect detected boxes from both results (raw coordinates on orig_frame)
            boxes = collect_boxes([res_bio, res_non])  # function from your script

            # 3) Save YOLO-format annotation (text) if requested
            if ANNOTATION_FORMAT in ("yolo", "both"):
                txt_path = save_yolo_annotation(base, boxes, orig_frame.shape)
                print("   -> YOLO annotation saved to", txt_path)

            # 4) Save Pascal VOC XML annotation if requested (and map names)
            if ANNOTATION_FORMAT in ("xml", "both"):
                xml_path = save_voc_xml(base, boxes, os.path.basename(img_path), orig_frame.shape)
                # optional: replace numeric <name> tags with model.names (if available)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        name_tag = obj.find("name")
                        if name_tag is not None:
                            try:
                                numeric = int(name_tag.text)
                                if isinstance(model.names, dict):
                                    name_tag.text = str(model.names.get(numeric, str(numeric)))
                                else:
                                    name_tag.text = str(model.names[numeric])
                            except Exception:
                                pass
                    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
                except Exception:
                    pass
                print("   -> VOC XML annotation saved to", xml_path)

finally:
    print("Releasing camera and windows.")
    cam.release()
    cv2.destroyAllWindows()
