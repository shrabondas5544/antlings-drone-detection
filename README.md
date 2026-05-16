# 🚁 Antlings Drone Detection System
### AI/ML Internship Technical Assessment — Human & Car Detection from Aerial Images

> **Stack:** Python · YOLOv8 · OpenCV · ByteTrack  
> **Tasks covered:** Dataset preprocessing · Model training · Detection & counting · Tracking (bonus) · Evaluation

---

## 📁 Project Structure

```
antlings-drone-detection/
├── src/
│   ├── preprocess.py          # Task-01: VisDrone → YOLO conversion + visualization
│   ├── train.py               # Task-02: YOLOv8 fine-tuning
│   ├── detect_and_count.py    # Task-03: Detection + human/car counting
│   ├── track.py               # Task-04 (Bonus): ByteTrack object tracking
│   └── evaluate.py            # Task-05: mAP / precision / recall evaluation
├── configs/
│   └── train_config.yaml      # All hyperparameters in one place
├── outputs/
│   ├── images/                # Processed still images
│   ├── videos/                # Processed videos
│   └── tracking/              # Tracked video outputs
├── requirements.txt
├── run_pipeline.sh            # One-command end-to-end runner
└── README.md
```

---

## ⚙️ Setup — Step by Step

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/antlings-drone-detection.git
cd antlings-drone-detection
```

### Step 2 — Create a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** If you have an NVIDIA GPU, install the matching CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> CPU-only also works — training just takes longer.

---

## 📦 Dataset Download — Step by Step

### Option A — Kaggle CLI (recommended, fastest)

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Get your API token:
#    Go to https://www.kaggle.com/settings → "Create New Token"
#    A kaggle.json file will be downloaded.

# 3. Place the token
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json   # Linux/macOS only

# 4. Download the dataset
kaggle datasets download -d banuprasadb/visdrone-dataset --unzip -p ./VisDrone
```

### Option B — Manual browser download

1. Open: <https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset>
2. Click **Download** (top-right). You will get `visdrone-dataset.zip`.
3. Unzip into the project folder:
   ```bash
   unzip visdrone-dataset.zip -d ./VisDrone
   ```

### Expected folder structure after download

```
VisDrone/
├── VisDrone2019-DET-train/
│   ├── images/       ← ~6,471 images
│   └── annotations/  ← one .txt per image
├── VisDrone2019-DET-val/
│   ├── images/       ← 548 images
│   └── annotations/
└── VisDrone2019-DET-test-dev/
    ├── images/       ← 1,610 images
    └── annotations/
```

---

## 🗂️ Task-01 — Dataset Understanding & Preprocessing

### What is VisDrone?

VisDrone is a large-scale benchmark collected by drones over various scenarios (urban, rural, crowd). It contains **10 object categories** — we extract only **person** (pedestrian + people) and **car** (car + van + truck + bus).

| Metric | Value |
|--------|-------|
| Training images | ~6,471 |
| Validation images | 548 |
| Avg objects / image | 54 |
| Small objects (<32px) | ~70% of all boxes |

### Key Challenges

- **Tiny objects** — people can be as small as 5×10 pixels
- **High density** — hundreds of objects per frame
- **Occlusion** — heavy overlap in crowd scenes
- **Lighting variance** — day, dusk, night conditions
- **Camera angle** — nadir and oblique views

### Run preprocessing

```bash
python src/preprocess.py \
    --visdrone-root ./VisDrone \
    --output-root   dataset_yolo \
    --visualize
```

This will:
1. Parse VisDrone annotations (CSV-style `.txt` files)
2. Filter to person & car classes only
3. Convert bounding boxes to YOLO format (normalized `cx cy w h`)
4. Copy images + write YOLO label files
5. Generate `dataset_yolo/dataset.yaml`
6. Save a sample grid visualization → `outputs/sample_viz.jpg`

### Augmentation strategy

The training script applies these augmentations, tuned for aerial imagery:

| Augmentation | Value | Reason |
|---|---|---|
| Mosaic | 1.0 | Exposes model to more object scales |
| Copy-paste | 0.1 | Synthesizes more crowded scenes |
| Mixup | 0.1 | Improves generalization |
| Horizontal flip | 0.5 | Aerial images are symmetric |
| Vertical flip | **0.0** | Disabled — sky/ground orientation matters |
| Rotation | **0.0** | Disabled — drone cameras are mostly nadir |
| HSV jitter | h=0.015, s=0.7, v=0.4 | Handles lighting variations |
| Scale | 0.5 | Simulates different flight altitudes |

---

## 🏋️ Task-02 — Model Training

### Why YOLOv8?

- State-of-the-art speed/accuracy tradeoff
- Native support for ByteTrack (tracking bonus)
- Built-in augmentation pipeline
- Anchor-free — handles irregular small objects well
- Easy to switch size (n/s/m/l/x) for hardware constraints

### Run training

```bash
python src/train.py \
    --data       dataset_yolo/dataset.yaml \
    --model-size s \
    --epochs     50  \
    --imgsz      640 \
    --batch      16  \
    --device     0
```

**Hardware guide:**

| GPU VRAM | Recommended batch | Model size |
|---|---|---|
| 4 GB | 8 | n |
| 8 GB | 16 | s |
| 16 GB | 32 | m |
| No GPU (CPU) | 4 | n |

> For CPU training, set `--device cpu --batch 4 --model-size n`

### Training outputs

```
runs/train/visdrone_exp/
├── weights/
│   ├── best.pt      ← use this for inference
│   └── last.pt
├── results.csv      ← loss and metric curves
├── confusion_matrix.png
├── PR_curve.png
└── val_batch*.jpg   ← sample predictions on val set
```

---

## 🎯 Task-03 — Detection & Human Counting

### Run on a single image

```bash
python src/detect_and_count.py \
    --source   path/to/drone_image.jpg \
    --weights  runs/train/visdrone_exp/weights/best.pt \
    --conf     0.35
```

### Run on a video

```bash
python src/detect_and_count.py \
    --source   path/to/drone_video.mp4 \
    --weights  runs/train/visdrone_exp/weights/best.pt \
    --conf     0.35
```

### What you will see

- 🟢 **Green boxes** = detected persons (with confidence score)
- 🔵 **Blue boxes** = detected cars (with confidence score)
- Top-left HUD overlay shows:
  - `Humans : N`
  - `Cars   : N`
  - `FPS    : N` (for video)

Results are saved to `outputs/images/` or `outputs/videos/`.

---

## 📍 Task-04 (Bonus) — Object Tracking

Uses **ByteTrack** (built into Ultralytics) — a fast, accurate multi-object tracker that associates detections across frames via IoU + confidence scores.

```bash
python src/track.py \
    --source   path/to/drone_video.mp4 \
    --weights  runs/train/visdrone_exp/weights/best.pt \
    --conf     0.35
```

### Tracking features

- Each object gets a **unique persistent ID** across frames
- Color is deterministically assigned per ID (same person = same color every frame)
- **Motion trails** drawn for last 30 frames
- Overlay shows both **frame count** and **cumulative unique count**

Output saved to `outputs/tracking/tracked_<filename>.mp4`

---

## 📊 Task-05 — Evaluation

```bash
python src/evaluate.py \
    --weights runs/train/visdrone_exp/weights/best.pt \
    --data    dataset_yolo/dataset.yaml \
    --split   val
```

### Typical results on VisDrone val (YOLOv8s, 50 epochs)

| Metric | Person | Car | Overall |
|--------|--------|-----|---------|
| Precision | ~0.62 | ~0.70 | ~0.66 |
| Recall | ~0.55 | ~0.65 | ~0.60 |
| mAP@0.5 | ~0.52 | ~0.65 | ~0.58 |
| mAP@0.5:0.95 | ~0.28 | ~0.40 | ~0.34 |

> Results vary by hardware, epochs, and random seed. Longer training with `yolov8m` gives better results.

---

## ⚡ One-Command Pipeline

```bash
bash run_pipeline.sh ./VisDrone path/to/test_image.jpg
```

Runs all 5 tasks end-to-end automatically.

---

## 💡 Strengths & Limitations

### Strengths
- YOLOv8 handles real-time inference efficiently
- Mosaic + copy-paste augmentation improves small object detection
- ByteTrack is robust to missed detections without requiring re-ID features
- Modular code — each task is an independent script

### Limitations
- Very small objects (<10px) still difficult to detect reliably
- Night / low-light performance degrades without IR-specific training data
- Counting is per-frame; for multi-camera scenarios, a more advanced method is needed
- ByteTrack may lose IDs during fast camera pans

### Ideas for improvement
- Use SAHI (Sliced Inference) for better small object recall
- Add night/augmented data via synthetic generation
- Implement StrongSORT or BoT-SORT for improved re-identification

---

## 🙏 Acknowledgements

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) — AISKYEYE Lab, Tianjin University
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
