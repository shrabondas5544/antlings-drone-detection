#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh  –  End-to-end pipeline: preprocess → train → detect → eval
# Usage:  bash run_pipeline.sh /path/to/VisDrone /path/to/test_image.jpg
# ─────────────────────────────────────────────────────────────────────────────

set -e

VISDRONE_ROOT="${1:-./VisDrone}"
TEST_SOURCE="${2:-}"   # image or video path for demo

DATASET_OUT="dataset_yolo"
WEIGHTS="runs/train/visdrone_exp/weights/best.pt"

echo "============================================"
echo "  Antlings Drone Detection Pipeline"
echo "============================================"

# ── Step 1: Preprocess (Skipped – Dataset already prepared) ───────────────────
# echo -e "\n[1/4] Preprocessing VisDrone dataset …"
# ./venv/Scripts/python.exe src/preprocess.py \
#     --visdrone-root "$VISDRONE_ROOT" \
#     --output-root   "$DATASET_OUT"  \
#     --visualize

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo -e "\n[2/4] Training YOLOv8s …"
./venv/Scripts/python.exe src/train.py \
    --data       "$DATASET_OUT/dataset.yaml" \
    --model-size s \
    --epochs     50  \
    --imgsz      640 \
    --batch      8  \
    --device     0 \
    --name       visdrone_exp \
    --resume

# ── Step 3: Detect (optional demo) ───────────────────────────────────────────
if [ -n "$TEST_SOURCE" ]; then
    echo -e "\n[3/4] Running detection on $TEST_SOURCE …"
    ./venv/Scripts/python.exe src/detect_and_count.py \
        --source   "$TEST_SOURCE" \
        --weights  "$WEIGHTS"     \
        --conf     0.35
else
    echo -e "\n[3/4] Skipping demo detection (no TEST_SOURCE provided)"
fi

# ── Step 4: Evaluate ──────────────────────────────────────────────────────────
echo -e "\n[4/4] Evaluating on validation set …"
./venv/Scripts/python.exe src/evaluate.py \
    --weights "$WEIGHTS" \
    --data    "$DATASET_OUT/dataset.yaml"

echo -e "\n✅  Pipeline complete! Check outputs/ for results."
