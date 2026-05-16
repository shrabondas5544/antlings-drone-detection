"""
Task-02: Model Training
Fine-tunes YOLOv8 on the VisDrone dataset for person & car detection.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str,
    model_size: str = "n",          # n / s / m / l / x
    epochs: int     = 50,
    imgsz: int      = 640,
    batch: int      = 16,
    device: str     = "0",          # "0" for GPU, "cpu" for CPU
    project: str    = "runs/train",
    name: str       = "visdrone_exp",
    resume: bool    = False,
):
    """
    Fine-tune YOLOv8 on VisDrone.

    Recommended starting point:
        python src/train.py \\
            --data dataset_yolo/dataset.yaml \\
            --model-size s \\
            --epochs 50 \\
            --imgsz 640 \\
            --batch 16 \\
            --device 0
    """
    if resume:
        checkpoint = Path(project) / name / "weights" / "last.pt"
        if checkpoint.exists():
            print(f"[RESUME] Loading checkpoint: {checkpoint}")
            model = YOLO(str(checkpoint))
        else:
            print(f"[WARNING] Checkpoint not found at {checkpoint}. Starting fresh.")
            model = YOLO(f"yolov8{model_size}.pt")
    else:
        weights = f"yolov8{model_size}.pt"
        print(f"[START] Loading base weights: {weights}")
        model = YOLO(weights)

    results = model.train(
        data      = data_yaml,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        device    = device,
        project   = project,
        name      = name,
        resume    = resume,
        # ── Augmentation hyperparams ──────────────────────────────────────
        hsv_h     = 0.015,   # hue shift
        hsv_s     = 0.7,     # saturation shift
        hsv_v     = 0.4,     # brightness shift
        degrees   = 0.0,     # rotation (keep 0 for aerial)
        translate = 0.1,
        scale     = 0.5,
        flipud    = 0.0,     # no vertical flip for aerial
        fliplr    = 0.5,
        mosaic    = 1.0,     # mosaic augmentation – great for small objects
        mixup     = 0.1,
        copy_paste= 0.1,
        # ── Optimizer ────────────────────────────────────────────────────
        optimizer = "SGD",
        lr0       = 0.01,
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        # ── Loss weights ─────────────────────────────────────────────────
        box       = 7.5,
        cls       = 0.5,
        dfl       = 1.5,
        # ── Misc ─────────────────────────────────────────────────────────
        workers   = 8,
        seed      = 42,
        exist_ok  = True,
        plots     = True,
        save      = True,
        save_period = 10,
        patience  = 20,
        val       = True,
        verbose   = True,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    print(f"\n[SUCCESS] Training complete!")
    print(f"   Best weights → {best_weights}")
    print(f"\n   Run detection:\n"
          f"   python src/detect_and_count.py --source <img/video> --weights {best_weights}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VisDrone")
    parser.add_argument("--data",       required=True, help="Path to dataset.yaml")
    parser.add_argument("--model-size", default="s",   choices=["n","s","m","l","x"])
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--batch",      type=int,   default=16)
    parser.add_argument("--device",     default="0")
    parser.add_argument("--project",    default="runs/train")
    parser.add_argument("--name",       default="visdrone_exp")
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    train(
        data_yaml   = args.data,
        model_size  = args.model_size,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        project     = args.project,
        name        = args.name,
        resume      = args.resume,
    )
