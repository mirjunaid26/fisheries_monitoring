# Fisheries Monitoring with SimpleFishNet 🐟

A streamlined Deep Learning solution for the [Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/competitions/the-nature-conservancy-fisheries-monitoring) challenge. This project implements a **Custom 5-Layer CNN (SimpleFishNet)** capable of simultaneous **Species Classification** and **Object Detection** (Bounding Box Regression).

## 🚀 Key Features
*   **SimpleFishNet**: A custom, lightweight CNN built from scratch (no massive pre-trained backbones).
*   **Dual-Head Architecture**:
    *   **Classification**: Identifies 8 fish species (ALB, BET, Sharks, etc.).
    *   **Detection**: Predicts bounding box coordinates for fish localization.
*   **Multi-Task Learning**: Trained with a weighted loss ($L_{cls} + \lambda L_{box}$).
*   **Community Data**: Utilizes official community-sourced bounding box annotations.
*   **Interactive Demo**: A CS231n-inspired web interface with a simulated edge-inference demo.

## 📂 Project Structure
*   `src/`: Core source code.
    *   `model.py`: `SimpleFishNet` definition (Shared backbone + 2 heads).
    *   `train.py`: Training loop with Multi-Task Loss.
    *   `dataset.py`: Custom dataset handling images and JSON bbox annotations.
    *   `download_bboxes.py`: Script to fetch annotation data.
    *   `export_onnx.py`: ONNX export utility.
*   `data/`: Data directory.
    *   `train/`: Training images.
    *   `bounding_boxes/`: JSON annotations.
*   `public/`: Website files (GitHub Pages).

## 🛠️ Usage

### 1. Setup
```bash
# Clone the repo
git clone https://github.com/mirjunaid26/fisheries_monitoring.git
cd fisheries_monitoring

# Install dependencies
pip install torch torchvision numpy pandas pillow tqdm requests onnx onnxscript
```

### 2. Prepare Data
Download the bounding box annotations:
```bash
python src/download_bboxes.py
```

### 3. Training
Train the SimpleFishNet (defaults to MPS/CUDA if available):
```bash
python -m src.train --epochs 10 --data_dir data/train
```

### 4. Export for Web
Generate the ONNX model for deployment:
```bash
python -m src.export_onnx
```

## 📊 Results (Proof-of-Concept)
*   **Architecture**: SimpleFishNet (custom)
*   **Val Classification Loss**: ~1.50
*   **Val Box MSE**: ~0.02
*   **Deployment**: Runs in real-time on Apple Silicon (MPS).

## 🌐 Website
Check out the [Project Website](https://mirjunaid26.github.io/fisheries_monitoring/) for a visual report and interactive demo.
