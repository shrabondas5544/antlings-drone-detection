# 🚁 Antlings Drone Detection & Tracking System
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch) ![YOLOv8](https://img.shields.io/badge/YOLOv8-State--of--the--Art-yellow) ![ByteTrack](https://img.shields.io/badge/ByteTrack-Tracking-green)

### AI/ML Engineering Internship Technical Assessment — Antlings

An end-to-end computer vision pipeline designed to detect, count, and track **Humans** and **Cars** from high-altitude aerial drone imagery. Built to handle the unique challenges of aerial datasets, including extreme scale variations, high object density, and complex occlusions.

---

## 📊 Executive Summary & Performance Metrics

This project fine-tunes a YOLOv8 architecture on the comprehensive VisDrone dataset. After rigorous training (50 epochs) with specialized data augmentation strategies tailored for nadir (top-down) views, the model achieves highly competitive results on the validation set (548 unseen images).

### Official Evaluation Results

| Metric | Overall Score | Person | Car |
| :--- | :---: | :---: | :---: |
| **Precision** | **77.9%** | 71.4% | 84.4% |
| **Recall** | **59.6%** | 42.6% | 76.7% |
| **mAP@0.50** | **65.7%** | 49.5% | **81.9%** |
| **mAP@0.50:0.95**| **37.6%** | 20.0% | 55.2% |

> **Data Insights:** The model demonstrates exceptional precision (**84.4%**) and mAP (**81.9%**) for detecting cars. Detecting humans from a drone perspective is notoriously difficult due to extreme low-pixel density (<10px bounding boxes), yet the model maintains a strong 71.4% precision rate, effectively minimizing false positives in crowded urban environments.

---

## 🛠️ Technical Architecture & Stack

*   **Core Framework:** PyTorch, Ultralytics YOLOv8s
*   **Tracking Algorithm:** ByteTrack (Multi-Object Tracking via IoU and confidence association)
*   **Data Processing:** OpenCV, Pandas, NumPy
*   **Hardware Profiling:** Optimized for NVIDIA RTX GPUs with CUDA/cuDNN acceleration.

---

## 🔬 Methodology & Pipeline Architecture

### 1. Dataset Preprocessing (Task 01)
The raw VisDrone dataset contains 10 categories. The preprocessing module (`preprocess.py`):
*   Parses CSV-style annotations and filters exclusively for `person` and `car` classes.
*   Converts coordinates from absolute bounding boxes to normalized YOLO format (`cx, cy, w, h`).
*   Generates a structured `dataset.yaml` for seamless framework integration.

### 2. Training Strategy & Augmentation (Task 02)
Aerial imagery requires specific augmentation strategies. Standard augmentations (like vertical flips) were **disabled** to preserve sky/ground orientation, while others were amplified:
*   **Mosaic (1.0) & Mixup (0.1):** Simulates dense, crowded environments.
*   **Scale (0.5):** Exposes the model to extreme altitude variations.
*   **HSV Jitter:** Ensures robustness across different lighting conditions (day, dusk, overcast).

### 3. Detection & Counting (Task 03)
The `detect_and_count.py` script runs real-time inference on static images or video feeds, rendering color-coded bounding boxes and generating a live On-Screen Display (OSD) HUD counting the current number of vehicles and pedestrians in the frame.

---

## 🌟 Bonus Feature: Real-Time Multi-Object Tracking (Task 04)

To elevate this project beyond static frame analysis, I implemented a temporal tracking pipeline using **ByteTrack**. 

**Key Features of the Tracking Module (`track.py`):**
*   **Persistent IDs:** Assigns a unique, deterministic ID and color to every detected entity.
*   **Motion Trails:** Renders historical trajectory paths (tails) for the last 30 frames to visualize movement patterns.
*   **Deduplication:** Maintains an absolute count of *unique* persons and cars that have entered the scene, preventing double-counting when objects become temporarily occluded.
*   **Real-Time UI:** Integrates `cv2.imshow` for live, real-time tracking visualization with an interactive early-exit protocol.

---

## 🚀 How to Run the Project

### Prerequisites
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the VisDrone dataset via Kaggle: `kaggle datasets download -d banuprasadb/visdrone-dataset`

### Quick Start Commands

**Run the Automated Pipeline:**
```bash
bash run_pipeline.sh ./VisDrone path/to/test_image.jpg
```

**Run Real-Time Tracking (Bonus):**
```bash
# Provide any drone .mp4 file
python src/track.py --source drone_sample.mp4 --weights runs/detect/runs/train/visdrone_exp/weights/best.pt --conf 0.35 --save-dir outputs/videos
```

**Run Evaluation:**
```bash
python src/evaluate.py --weights runs/detect/runs/train/visdrone_exp/weights/best.pt --data dataset_yolo/dataset.yaml --split val
```

---

## 📈 Future Scalability & Improvements
If deployed to production, the following architectural enhancements would be introduced:
1.  **SAHI (Slicing Aided Hyper Inference):** Implementing sliding window inference to drastically improve recall on micro-objects (humans < 5px).
2.  **Re-Identification (ReID):** Upgrading from ByteTrack to BoT-SORT to maintain track IDs across longer occlusions or multi-camera setups.
3.  **TensorRT Export:** Compiling the `.pt` weights to `.engine` for sub-10ms edge inference on companion computers (e.g., Jetson Nano/Orin) mounted directly on the drones.

---
*Developed for the Antlings AI/ML Engineering Technical Assessment.*
