"""
Antlings Internship – AI/ML Drone Human Detection & Counting System
Task-03: Human & Car Detection with Human Counting
Author: [Your Name]
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import time
from ultralytics import YOLO


# ─── Class Indices (VisDrone → COCO-based fine-tuned model) ───────────────────
PERSON_CLASSES = {0}          # 'person'
CAR_CLASSES    = {1}          # 'car'

CLASS_COLORS = {
    "person": (0, 255, 100),   # green
    "car":    (0, 120, 255),   # orange-blue
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bounding_box(img, box, label, color, conf):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), FONT, 0.5, (255, 255, 255), 1)


def draw_overlay(img, human_count, car_count, fps=None):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (260, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, f"Humans : {human_count}", (20, 40),  FONT, 0.8, (0, 255, 100), 2)
    cv2.putText(img, f"Cars   : {car_count}",   (20, 75),  FONT, 0.8, (0, 120, 255), 2)
    if fps:
        cv2.putText(img, f"FPS    : {fps:.1f}", (20, 105), FONT, 0.6, (200, 200, 200), 1)

    # Watermark
    cv2.putText(img, "Antlings Drone Vision", (w - 230, h - 12),
                FONT, 0.5, (180, 180, 180), 1)


def process_image(model, img_path: str, conf_thresh=0.35, save_dir="outputs/images"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {img_path}")

    results = model(img, conf=conf_thresh, verbose=False)[0]
    human_count = 0
    car_count   = 0

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        if cls in PERSON_CLASSES:
            human_count += 1
            draw_bounding_box(img, xyxy, "person", CLASS_COLORS["person"], conf)
        elif cls in CAR_CLASSES:
            car_count += 1
            draw_bounding_box(img, xyxy, "car", CLASS_COLORS["car"], conf)

    draw_overlay(img, human_count, car_count)

    out_path = Path(save_dir) / Path(img_path).name
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[IMAGE] Humans={human_count}  Cars={car_count}  → {out_path}")
    return human_count, car_count


def process_video(model, vid_path: str, conf_thresh=0.35, save_dir="outputs/videos"):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {vid_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / Path(vid_path).name
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_thresh, verbose=False)[0]
        human_count = 0
        car_count   = 0

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            if cls in PERSON_CLASSES:
                human_count += 1
                draw_bounding_box(frame, xyxy, "person", CLASS_COLORS["person"], conf)
            elif cls in CAR_CLASSES:
                car_count += 1
                draw_bounding_box(frame, xyxy, "car", CLASS_COLORS["car"], conf)

        elapsed = time.time() - t0
        draw_overlay(frame, human_count, car_count, fps=1.0 / (elapsed + 1e-9))

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: humans={human_count}  cars={car_count}")

    cap.release()
    writer.release()
    print(f"[VIDEO] Saved → {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Human & Car Detection")
    parser.add_argument("--source",  required=True, help="Image or video path")
    parser.add_argument("--weights", default="runs/train/exp/weights/best.pt",
                        help="Path to trained YOLO weights")
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--save-dir", default="outputs")
    args = parser.parse_args()

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    src = args.source
    if Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        process_image(model, src, args.conf, f"{args.save_dir}/images")
    else:
        process_video(model, src, args.conf, f"{args.save_dir}/videos")
