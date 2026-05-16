"""
Task-04 (Bonus): Object Tracking using YOLO built-in ByteTrack
Tracks persons and cars across frames with unique IDs.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import time
from collections import defaultdict
from ultralytics import YOLO

PERSON_CLASSES = {0}
CAR_CLASSES    = {1}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Track history for drawing trails
track_history = defaultdict(list)


def random_color(track_id: int):
    """Deterministic color per track ID."""
    np.random.seed(track_id * 31 + 7)
    return tuple(int(c) for c in np.random.randint(80, 255, 3))


def draw_tracked_box(img, box, track_id, label, conf):
    x1, y1, x2, y2 = map(int, box)
    color = random_color(track_id)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} #{track_id} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), FONT, 0.5, (255, 255, 255), 1)

    # Center point for trail
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    track_history[track_id].append((cx, cy))
    if len(track_history[track_id]) > 30:
        track_history[track_id].pop(0)

    # Draw trail
    pts = track_history[track_id]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        thickness = max(1, int(alpha * 3))
        cv2.line(img, pts[i-1], pts[i], color, thickness)


def draw_overlay(img, human_count, car_count, unique_persons, unique_cars, fps=None):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (310, 135), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    cv2.putText(img, f"Persons (frame) : {human_count}", (20, 40),  FONT, 0.7, (0, 255, 100), 2)
    cv2.putText(img, f"Cars    (frame) : {car_count}",   (20, 70),  FONT, 0.7, (0, 120, 255), 2)
    cv2.putText(img, f"Unique persons  : {unique_persons}", (20, 100), FONT, 0.7, (100, 255, 200), 1)
    cv2.putText(img, f"Unique cars     : {unique_cars}",   (20, 125), FONT, 0.7, (100, 180, 255), 1)
    if fps:
        cv2.putText(img, f"FPS: {fps:.1f}", (w - 100, 30), FONT, 0.7, (200, 200, 200), 2)
    cv2.putText(img, "Antlings Drone Tracker", (w - 250, h - 12), FONT, 0.5, (180, 180, 180), 1)


def track_video(model, vid_path: str, conf_thresh=0.35, save_dir="outputs/tracking"):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {vid_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / ("tracked_" + Path(vid_path).name)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    unique_person_ids = set()
    unique_car_ids    = set()
    frame_idx = 0

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tracking (ByteTrack built-in)
        results = model.track(
            frame,
            conf=conf_thresh,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        human_count = 0
        car_count   = 0

        if results.boxes.id is not None:
            for box, track_id, cls, conf in zip(
                results.boxes.xyxy,
                results.boxes.id.int().tolist(),
                results.boxes.cls.int().tolist(),
                results.boxes.conf.tolist()
            ):
                xyxy = box.tolist()
                if cls in PERSON_CLASSES:
                    human_count += 1
                    unique_person_ids.add(track_id)
                    draw_tracked_box(frame, xyxy, track_id, "person", conf)
                elif cls in CAR_CLASSES:
                    car_count += 1
                    unique_car_ids.add(track_id)
                    draw_tracked_box(frame, xyxy, track_id, "car", conf)

        elapsed = time.time() - t0
        draw_overlay(frame, human_count, car_count,
                     len(unique_person_ids), len(unique_car_ids),
                     fps=1.0 / (elapsed + 1e-9))

        writer.write(frame)
        
        # Real-time view
        cv2.imshow("Antlings Tracker (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tracking stopped by user.")
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: persons={human_count} cars={car_count} "
                  f"unique_persons={len(unique_person_ids)} unique_cars={len(unique_car_ids)}")

    cap.release()
    cv2.destroyAllWindows()
    writer.release()
    print(f"\n✅ Tracking complete!")
    print(f"   Total unique persons tracked : {len(unique_person_ids)}")
    print(f"   Total unique cars tracked    : {len(unique_car_ids)}")
    print(f"   Saved → {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Object Tracker (ByteTrack)")
    parser.add_argument("--source",  required=True, help="Video path")
    parser.add_argument("--weights", default="runs/train/visdrone_exp/weights/best.pt")
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--save-dir", default="outputs/tracking")
    args = parser.parse_args()

    model = YOLO(args.weights)
    track_video(model, args.source, args.conf, args.save_dir)
