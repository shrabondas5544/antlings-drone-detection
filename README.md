# 🚁 Antlings Drone Detection & Tracking System

## 📖 High‑Level Overview

This repository implements a **complete end‑to‑end computer‑vision pipeline** for detecting, counting, and tracking **people** and **vehicles** from aerial drone imagery. The system is built around the state‑of‑the‑art **YOLOv8** object detector and the **ByteTrack** multi‑object tracker, both from the Ultralytics ecosystem.

At a glance the workflow consists of:
1. **Data acquisition & preprocessing** – converting the VisDrone benchmark into YOLO‑compatible format, selecting only the `person` and `car` classes.
2. **Model training** – fine‑tuning YOLOv8 (size *s*) on the curated dataset with custom augmentations that reflect aerial‑view challenges (mosaic, copy‑paste, scale, HSV jitter).
3. **Inference & counting** – running the trained model on single images or video streams, showing live bounding‑box overlays and per‑frame object counts.
4. **Real‑time multi‑object tracking (bonus)** – using ByteTrack to assign persistent IDs, draw motion trails, and keep a *unique* count of objects across the whole video.
5. **Evaluation** – computing standard detection metrics (precision, recall, mAP@0.5, mAP@0.5:0.95) on a held‑out validation set.

The pipeline can be executed with a **single Bash command** (`run_pipeline.sh`) or step‑by‑step for educational purposes.

---

## 📁 Repository Layout

```
antlings-drone-detection/
├── src/                     # All Python scripts (preprocess, train, detect, track, evaluate)
│   ├── preprocess.py        # Task‑01 – VisDrone → YOLO conversion + visual sanity‑check
│   ├── train.py             # Task‑02 – YOLOv8 fine‑tuning
│   ├── detect_and_count.py  # Task‑03 – Inference + per‑frame counting HUD
│   ├── track.py             # Task‑04 – ByteTrack based multi‑object tracking (bonus)
│   └── evaluate.py          # Task‑05 – Model evaluation (precision, recall, mAP)
├── configs/                 # Hyper‑parameter YAML (train_config.yaml)
├── outputs/                 # Generated artefacts (images, videos, tracking results, eval summary)
│   ├── images/              # Sample detection images (saved by detect_and_count.py)
│   ├── videos/              # Sample detection videos (if any)
│   └── tracking/            # Tracked video outputs (bonus task)
├── dataset_yolo/            # YOLO‑style dataset generated from VisDrone (auto‑created)
│   ├── images/              # train/val/test image folders
│   ├── labels/              # Corresponding .txt label files
│   └── dataset.yaml         # Dataset config used by Ultralytics
├── run_pipeline.sh          # One‑command orchestrator (preprocess → train → demo)
├── requirements.txt         # Exact Python dependencies
├── README.md                # **THIS FILE** – full guide for users & recruiters
└── .gitignore               # Keeps large binaries & virtual env out of VCS
```

---

## ⚙️ Setup – Step‑by‑Step for a Normal User

### 1️⃣ Clone the Repository
```bash
# HTTPS clone (replace with SSH if you prefer)
git clone https://github.com/shrabondas5544/antlings-drone-detection.git
cd antlings-drone-detection
```

### 2️⃣ Create & Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux (for reference)
python3 -m venv venv
source venv/bin/activate
```
> A virtual environment isolates the required packages from your system Python, preventing version conflicts.

### 3️⃣ Install Python Dependencies
```bash
pip install -r requirements.txt
```
> The `requirements.txt` pins versions of `torch`, `ultralytics`, `opencv‑python`, and other utilities needed for reproducibility.

### 4️⃣ (Optional) Install GPU‑accelerated PyTorch
If you have an NVIDIA GPU with CUDA 12.1 (or newer) you can get a huge speed‑up:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Otherwise the CPU fallback will work, albeit slower.

### 5️⃣ Download the VisDrone Dataset
The raw VisDrone dataset is ~15 GB; use the Kaggle CLI for a quick download.
```bash
pip install kaggle   # if not already installed
# Authenticate with your Kaggle API token (downloaded from kaggle.com → My Account)
mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d banuprasadb/visdrone-dataset --unzip -p ./VisDrone
```
After extraction you should see the following folder tree (simplified):
```
VisDrone/                      # Root of the raw dataset
├── VisDrone2019-DET-train/   # 6,471 training images + annotations
├── VisDrone2019-DET-val/     # 548 validation images + annotations
└── VisDrone2019-DET-test-dev/ # 1,610 test images (optional)
```

---

## 🧩 Detailed Process Flow (What Happens Under the Hood?)

### 1️⃣ Dataset Preprocessing (Task‑01)
`src/preprocess.py` performs the following operations:
- **Parse annotations** (`.txt` files) that contain 10 object categories.
- **Filter classes** – keep only `person` (ID 0) and `car` (ID 1). All other categories are discarded.
- **Convert coordinates** from absolute pixel values to YOLO normalized format `(cx, cy, w, h)` where each value lies in `[0, 1]`.
- **Copy images** into a clean directory hierarchy (`dataset_yolo/images/{train,val,test}`) and write matching label files (`*.txt`).
- **Generate `dataset.yaml`** – a tiny YAML file telling Ultralytics where the images and labels are, and the class names.
- **Create a visual sanity‑check** (`outputs/sample_viz.jpg`) that mosaics a few random samples with their bounding boxes, helping you verify the conversion.

Running it is as simple as:
```bash
python src/preprocess.py \
    --visdrone-root ./VisDrone \
    --output-root   dataset_yolo \
    --visualize
```
You will see a progress bar and the final `sample_viz.jpg`.

---

### 2️⃣ Model Training (Task‑02)
`src/train.py` wraps the Ultralytics trainer with a custom config (`configs/train_config.yaml`). Key points:
- **Model size:** `s` (YOLOv8‑small) – a sweet spot for 8 GB VRAM GPUs.
- **Epochs:** 50 – enough to converge on VisDrone while keeping training time reasonable (≈1 hour on RTX 4060).
- **Image size:** 640 px – balances speed and accuracy for aerial imagery.
- **Batch size:** 16 – matches the 8 GB VRAM recommendation.
- **Custom augmentations** (defined in `train_config.yaml`):
  - **Mosaic (1.0)** – stitches 4 random images together, exposing the model to varied scales.
  - **Copy‑Paste (0.1)** – inserts additional objects to simulate crowded scenes.
  - **Scale (0.5)** – mimics different flight altitudes.
  - **HSV jitter (h=0.015, s=0.7, v=0.4)** – helps the model generalise across lighting conditions.
- **Optimizer:** SGD with momentum 0.937 – proven stable for YOLOv8 training.

Launch training with:
```bash
python src/train.py \
    --data dataset_yolo/dataset.yaml \
    --model-size s \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --device 0
```
All checkpoints (`epoch0.pt`, `epoch10.pt`, …) and the final **`best.pt`** are saved under `runs/train/visdrone_exp/weights/`.

---

### 3️⃣ Detection & Counting (Task‑03)
`src/detect_and_count.py` runs inference on **any** image or video source (`0` for webcam, a file path, or a YouTube URL). It:
1. Loads `best.pt`.
2. Runs the model frame‑by‑frame.
3. Draws **green** boxes for people and **blue** boxes for cars.
4. Overlays a HUD that shows:
   - `Persons (frame) : N`
   - `Cars    (frame) : N`
   - Optionally the FPS.
5. Saves the annotated image/video to `outputs/images/` or `outputs/videos/`.

Example commands:
```bash
# Single image
python src/detect_and_count.py \
    --source dataset_yolo/images/val/0000001_02999_d_0000005.jpg \
    --weights runs/detect/runs/train/visdrone_exp/weights/best.pt \
    --conf 0.35

# Video (or webcam)
python src/detect_and_count.py \
    --source path/to/drone_video.mp4 \
    --weights runs/detect/runs/train/visdrone_exp/weights/best.pt \
    --conf 0.35
```
The script prints a short summary (`Humans=## Cars=##`) and writes the output file.

---

### 4️⃣ Bonus – Real‑Time Multi‑Object Tracking (Task‑04)
The `track.py` script is a **first‑class** addition that turns per‑frame detections into *temporal tracks*.

#### How it works internally
1. **ByteTrack integration** – Ultralytics provides a built‑in `model.track()` method that implements the ByteTrack algorithm (high‑performance IoU‑based association).
2. **Persistent IDs** – Each detection receives an integer `track_id`. The script seeds a deterministic colour palette so the same ID always appears in the same colour.
3. **Motion trails** – The script stores the recent centre points of each track (max 30 frames) and draws semi‑transparent line segments, visualising the object's path.
4. **Unique counting** – Two Python `set`s (`unique_person_ids`, `unique_car_ids`) collect IDs the first time they appear, allowing you to report *total distinct* people and cars seen in the whole video.
5. **Live preview** – `cv2.imshow()` displays the processed video frame‑by‑frame; press **`q`** to abort early.
6. **Output video** – The processed video (with boxes, IDs, trails, and HUD) is saved to the directory you specify with `--save-dir` (default `outputs/tracking`).

#### Running the tracker
```bash
python src/track.py \
    --source drone_sample.mp4 \
    --weights runs/detect/runs/train/visdrone_exp/weights/best.pt \
    --conf 0.35 \
    --save-dir outputs/videos
```
After completion you’ll see a console summary, e.g.:
```
✅ Tracking complete!
   Total unique persons tracked : 5
   Total unique cars tracked    : 3
   Saved → outputs/videos/tracked_drone_sample.mp4
```

---

### 5️⃣ Evaluation (Task‑05)
`src/evaluate.py` loads the validation split (`dataset_yolo/dataset.yaml` → `val`) and computes the standard COCO metrics using Ultralytics’ built‑in evaluator.
```bash
python src/evaluate.py \
    --weights runs/detect/runs/train/visdrone_exp/weights/best.pt \
    --data dataset_yolo/dataset.yaml \
    --split val
```
The script prints a concise table and writes `outputs/eval_summary.json` (machine‑readable) and a human‑readable summary in the console.

#### Official numbers (as of the latest run)
| Metric | Overall | Person | Car |
|---|---|---|---|
| **Precision** | **77.9 %** | 71.4 % | **84.4 %** |
| **Recall** | **59.6 %** | 42.6 % | 76.7 % |
| **mAP@0.5** | **65.7 %** | 49.5 % | **81.9 %** |
| **mAP@0.5:0.95** | **37.6 %** | 20.0 % | 55.2 % |

These numbers are stored in `outputs/eval_summary.json` for easy inclusion in reports.

---

## 🎬 One‑Command End‑to‑End Pipeline
The repository ships with `run_pipeline.sh`, a thin Bash wrapper that runs the full workflow with a single line:
```bash
bash run_pipeline.sh ./VisDrone path/to/test_image.jpg
```
What it does:
1. Calls `preprocess.py` → creates `dataset_yolo/`.
2. Calls `train.py` → fine‑tunes the model.
3. Calls `detect_and_count.py` on the supplied test image (or video) and saves the result.
4. (Optional) Calls `track.py` if a video source is detected.
5. Calls `evaluate.py` and prints the final metrics.

---

## 🛠️ Troubleshooting & Tips for Normal Users
| Symptom | Likely Cause | Quick Fix |
|---|---|---|
| **“CUDA out of memory”** | Batch size too large for GPU VRAM. | Reduce `batch` in `train_config.yaml` (e.g., from 16 → 8) or switch to `model-size n`. |
| **No detections** | Confidence threshold too high or model not trained. | Lower `--conf` (e.g., `0.25`) and confirm that `best.pt` exists in `runs/detect/.../weights/`. |
| **Tracking ID jumps** | ByteTrack parameters not tuned for very fast motion. | Increase `tracker` confidence (`--conf`) or add `--track-buffer 30` (default is 30). |
| **Slow inference on CPU** | Using CPU instead of GPU. | Install CUDA‑enabled PyTorch as shown in the *GPU note* above. |
| **Dataset conversion errors** | Original VisDrone annotations missing or corrupted. | Re‑download the dataset, verify the `VisDrone` folder structure, and run `preprocess.py` again. |

---

## 📚 Additional Resources
- **VisDrone Dataset** – a large‑scale benchmark for aerial object detection: https://github.com/VisDrone/VisDrone-Dataset
- **Ultralytics YOLOv8** – the underlying detection framework: https://github.com/ultralytics/ultralytics
- **ByteTrack** – high‑performance multi‑object tracker: https://github.com/ifzhang/ByteTrack
- **SAHI (Sliced Inference)** – useful for tiny object recall: https://github.com/obss/sahi
- **BoT‑SORT** – a tracking algorithm with Re‑ID features for longer occlusions: https://github.com/NirAharon/BoT‑SORT

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an Issue or Pull Request if you:
- Find a bug in the preprocessing script.
- Want to add a new augmentation (e.g., random rotation for non‑nadir drones).
- Implement a more sophisticated tracker (e.g., StrongSORT).

Please keep the `.gitignore` updated to avoid committing large binary files.

---

## 📄 License
This project is released under the **MIT License** – you are free to use, modify, and distribute it for both academic and commercial purposes.

---

*Developed for the Antlings AI/ML Engineering Technical Assessment.*
