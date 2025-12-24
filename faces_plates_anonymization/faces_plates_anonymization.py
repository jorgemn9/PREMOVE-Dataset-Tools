# python blur_plates_faces.py   --source input   --out output   --face-weights ./weights/yolo11n_faces.pt   --plate-weights ./weights/yolo11n_licenseplates.pt --overwrite

"""
Script: blur_plates_faces.py
Description: Detects FACES and LICENSE PLATES in images or videos using separate YOLO models
             and pixelates (mosaic) those regions for anonymization.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ============================
# Ultralytics / Torch safe load
# ============================
try:
    import torch
    from torch.serialization import add_safe_globals
    try:
        from ultralytics.nn.tasks import DetectionModel
        add_safe_globals([DetectionModel])
    except Exception:
        pass
except Exception:
    pass

try:
    from ultralytics import YOLO
except Exception:
    print("[ERROR] Install ultralytics: pip install ultralytics", file=sys.stderr)
    raise

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


# ============================
# CLI
# ============================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Blur faces and license plates (keep folder structure)")
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="./out")
    p.add_argument("--face-weights", required=True)
    p.add_argument("--plate-weights", required=True)
    p.add_argument("--face-conf", type=float, default=0.15)
    p.add_argument("--plate-conf", type=float, default=0.04)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="auto")
    p.add_argument("--margin", type=float, default=0.12)
    p.add_argument("--mosaic-size", type=int, default=6)
    p.add_argument("--blur", action="store_true")
    p.add_argument("--blur-ksize", type=int, default=35)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--draw-rects", action="store_true")
    return p.parse_args()


# ============================
# Utils
# ============================
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_models(face_w: str, plate_w: str) -> Tuple[YOLO, YOLO]:
    return YOLO(face_w), YOLO(plate_w)


def expand_box(xyxy, w, h, margin):
    x1, y1, x2, y2 = xyxy
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w - 1, x2 + dx),
        min(h - 1, y2 + dy),
    )


def mosaic_region(img, box, block):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    roi_small = cv2.resize(roi, (max(1, w // block), max(1, h // block)))
    roi_mosaic = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = roi_mosaic


def blur_region(img, box, k):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    if k % 2 == 0:
        k += 1
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def collect_boxes(results, shape):
    boxes = []
    h, w = shape[:2]
    if not results or results[0].boxes is None:
        return boxes
    for b in results[0].boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = b
        if x2 > x1 and y2 > y1:
            boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return boxes


def redact_frame(frame, face_model, plate_model, args):
    h, w = frame.shape[:2]

    faces = face_model.predict(frame, conf=args.face_conf, iou=args.iou, device=args.device, verbose=False)
    plates = plate_model.predict(frame, conf=args.plate_conf, iou=args.iou, device=args.device, verbose=False)

    boxes = collect_boxes(faces, frame.shape) + collect_boxes(plates, frame.shape)

    for b in boxes:
        x1, y1, x2, y2 = expand_box(b, w, h, args.margin)
        if args.blur:
            blur_region(frame, (x1, y1, x2, y2), args.blur_ksize)
        else:
            mosaic_region(frame, (x1, y1, x2, y2), args.mosaic_size)
        if args.draw_rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


# ============================
# Processing
# ============================
def process_image(in_path, out_path, face_model, plate_model, args):
    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] {out_path}")
        return
    img = cv2.imread(str(in_path))
    if img is None:
        print(f"[WARN] Cannot read {in_path}")
        return
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), redact_frame(img, face_model, plate_model, args))
    print(f"[OK] {out_path}")


def process_video(in_path, out_path, face_model, plate_model, args):
    ensure_dir(out_path.parent)
    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(redact_frame(frame, face_model, plate_model, args))

    cap.release()
    writer.release()
    print(f"[OK] {out_path}")


def iter_sources(src: Path):
    if src.is_dir():
        yield from [p for p in src.rglob("*") if p.is_file() and (is_image(p) or is_video(p))]
    else:
        yield src


# ============================
# Main
# ============================
def main():
    args = parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    src = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    print("[INFO] Loading models...")
    face_model, plate_model = load_models(args.face_weights, args.plate_weights)
    print("[INFO] Ready")

    for path in iter_sources(src):
        rel = path.relative_to(src)
        out_path = out_root / rel

        if is_image(path):
            process_image(path, out_path, face_model, plate_model, args)
        elif is_video(path):
            out_video = out_path.with_stem(out_path.stem + "_redacted")
            process_video(path, out_video, face_model, plate_model, args)


if __name__ == "__main__":
    main()