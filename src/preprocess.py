"""
Task-01: Dataset Understanding & Preprocessing
Converts VisDrone annotations → YOLO format and applies augmentation strategy.
"""

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── VisDrone class IDs we care about ────────────────────────────────────────
# VisDrone: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car,
#           5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others

YOLO_TO_YOLO = {
    0: 0,   # pedestrian  -> person
    1: 0,   # people      -> person
    3: 1,   # car         -> car
    4: 1,   # van         -> car
    5: 1,   # truck       -> car
    8: 1,   # bus         -> car
}

CLASS_NAMES = ["person", "car"]


def convert_visdrone_annotation(ann_path: Path):
    """
    Reads YOLO format labels and maps classes to 0 (person) and 1 (car).
    """
    yolo_lines = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cat_id = int(parts[0])

            if cat_id not in YOLO_TO_YOLO:
                continue

            yolo_cls = YOLO_TO_YOLO[cat_id]
            yolo_lines.append(f"{yolo_cls} {' '.join(parts[1:])}")
    return yolo_lines


def build_yolo_dataset(visdrone_root: str, output_root: str, splits=("train", "val", "test")):
    """
    Reads VisDrone folder structure and writes YOLO-compatible dataset.

    VisDrone expected layout:
        <visdrone_root>/
            VisDrone2019-DET-train/
                images/*.jpg
                labels/*.txt
            VisDrone2019-DET-val/
                images/*.jpg
                labels/*.txt
    """
    visdrone_root = Path(visdrone_root)
    output_root   = Path(output_root)

    split_map = {
        "train": "VisDrone2019-DET-train",
        "val":   "VisDrone2019-DET-val",
        "test":  "VisDrone2019-DET-test-dev",
    }

    skipped = 0
    total   = 0

    for split in splits:
        folder = split_map.get(split)
        if not folder:
            continue

        img_dir = visdrone_root / folder / "images"
        ann_dir = visdrone_root / folder / "labels"

        out_img = output_root / "images" / split
        out_lbl = output_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"\n[{split.upper()}] Processing {len(images)} images …")

        for img_path in tqdm(images):
            ann_path = ann_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            lines = convert_visdrone_annotation(ann_path)
            if not lines:
                skipped += 1
                continue

            # Copy image
            shutil.copy(img_path, out_img / img_path.name)
            # Write label
            with open(out_lbl / (img_path.stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            total += 1

    print(f"\n[SUCCESS] Done. Total processed: {total}  Skipped: {skipped}")
    _write_yaml(output_root)


def _write_yaml(output_root: Path):
    yaml_content = f"""# VisDrone -> YOLO dataset config
path: {output_root.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 2
names: ['person', 'car']
"""
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[FILE] Dataset YAML written -> {yaml_path}")


def visualize_samples(dataset_root: str, num_samples: int = 6, save_path: str = "outputs/sample_viz.jpg"):
    """Draw GT boxes on random training samples and save a grid."""
    dataset_root = Path(dataset_root)
    images = list((dataset_root / "images" / "train").glob("*.jpg"))
    random.shuffle(images)
    images = images[:num_samples]

    palette = [(0, 255, 100), (0, 120, 255)]
    panels  = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = dataset_root / "labels" / "train" / (img_path.stem + ".txt")

        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = palette[cls % len(palette)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, CLASS_NAMES[cls], (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Resize to uniform height
        target_h = 320
        scale = target_h / h
        img = cv2.resize(img, (int(w * scale), target_h))
        panels.append(img)

    if not panels:
        print("No panels to display.")
        return

    # Build 2-row grid
    mid  = len(panels) // 2
    row1 = np.hstack(panels[:mid]) if mid else panels[0]
    row2 = np.hstack(panels[mid:]) if panels[mid:] else None

    if row2 is not None:
        # Pad widths to match
        if row1.shape[1] != row2.shape[1]:
            max_w = max(row1.shape[1], row2.shape[1])
            def pad(img, target_w):
                pad_w = target_w - img.shape[1]
                return np.pad(img, ((0,0),(0,pad_w),(0,0)), constant_values=30)
            row1 = pad(row1, max_w)
            row2 = pad(row2, max_w)
        grid = np.vstack([row1, row2])
    else:
        grid = row1

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, grid)
    print(f"[IMAGE] Sample visualization saved -> {save_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visdrone-root", required=True, help="Path to VisDrone root folder")
    parser.add_argument("--output-root",   default="dataset_yolo")
    parser.add_argument("--visualize",     action="store_true")
    args = parser.parse_args()

    build_yolo_dataset(args.visdrone_root, args.output_root)
    if args.visualize:
        visualize_samples(args.output_root)
