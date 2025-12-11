# Fisheries Monitoring with Advanced Deep Learning

[![Project Website](https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=github)](https://mirjunaid26.github.io/fisheries_monitoring/)


This project implements state-of-the-art computer vision models to detect and classify fish species from fishing boat camera footage.

## 🧠 Architectures

The project features two advanced architectures located in `src/models/`:

1.  **CNN (ResNet50)** (`src/models/cnn.py`):
    - Uses a Pre-trained ResNet50 backbone.
    - Custom heads for **Classification** (8 classes) and **Bounding Box Regression**.
    - Robust and distinct feature extraction.

2.  **Transformer (ViT)** (`src/models/transformer.py`):
    - Uses a Vision Transformer (`vit_base_patch16_224`) backbone.
    - Leverages self-attention mechanisms for global context.
    - Fine-tuned for simultaneous classification and localization.

## 📂 Project Structure

```
├── data/               # Dataset (Train images and JSON annotations)
├── app.py              # Gradio Web Application
├── src/
│   ├── train.py        # Main Training Script
│   ├── models/         # Neural Architectures
│   │   ├── cnn.py      # ResNet50 Implementation
│   │   └── transformer.py # Vision Transformer Implementation
│   └── ...
└── requirements.txt    # Dependencies
```

## 🚀 Getting Started

### 1. Installation

```bash
conda create -n fisheries python=3.10
conda activate fisheries
pip install -r requirements.txt
```

### 2. Training

Train your preferred model architecture.

**Train CNN (ResNet):**
```bash
python src/train.py --model_type cnn --epochs 10 --learning_rate 1e-4
```

**Train Transformer (ViT):**
```bash
python src/train.py --model_type vit --epochs 10 --learning_rate 5e-5
```

### 3. Web Application (Demo)

Launch the interactive web interface to test the model.

```bash
python app.py --model_type cnn --model_path src/models/best_model_cnn.pth
```
*Note: Make sure to select the architecture that matches your trained model.*

## 🌐 Publishing

This project uses **Gradio** for the web interface and can be automatically deployed to **Hugging Face Spaces** using the included GitHub Action.

### Automated Deployment (GitHub Actions)
1.  Create a **New Space** on Hugging Face (e.g., `fisheries_monitoring`).
2.  In your GitHub Repository Settings -> Secrets and variables -> Actions:
    - Create a New Repository Secret named `HF_TOKEN` with your Hugging Face Access Token (Write permissions).
3.  Edit `.github/workflows/sync_to_hub.yml` to match your Hugging Face username and space name if they differ from the default.
4.  Push to `main`, and the action will automatically sync your code to the Space!

### Manual Deployment
1. Create a Space on Hugging Face.
2. Upload `app.py`, `src/`, and `requirements.txt`.
