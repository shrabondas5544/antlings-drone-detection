"""
Task-05: Evaluation & Visualization
Computes mAP, precision, recall and generates a visual report.
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def evaluate(weights: str, data_yaml: str, imgsz: int = 640, split: str = "val"):
    """
    Run YOLO validation and print metrics.
    """
    model = YOLO(weights)
    metrics = model.val(
        data    = data_yaml,
        imgsz   = imgsz,
        split   = split,
        verbose = True,
        plots   = True,
        save_json = True,
    )

    print("\n" + "="*55)
    print("  EVALUATION RESULTS")
    print("="*55)
    print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")
    print("="*55)

    # Per-class
    class_names = ["person", "car"]
    for i, name in enumerate(class_names):
        try:
            print(f"  [{name:<8}] AP50={metrics.box.ap50[i]:.4f}  "
                  f"AP={metrics.box.ap[i]:.4f}")
        except (IndexError, AttributeError):
            pass
    print("="*55)

    # Save summary JSON
    summary = {
        "mAP50":      float(metrics.box.map50),
        "mAP50_95":   float(metrics.box.map),
        "precision":  float(metrics.box.mp),
        "recall":     float(metrics.box.mr),
    }
    out = Path("outputs/eval_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n📄 Summary saved → {out}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model")
    parser.add_argument("--weights",  required=True)
    parser.add_argument("--data",     required=True, help="Path to dataset.yaml")
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--split",    default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.imgsz, args.split)
