# Fisheries Monitoring with Advanced Deep Learning

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
├── src/
│   ├── app.py          # Gradio Web Application
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
python src/app.py --model_type cnn --model_path src/models/best_model_cnn.pth
```

## 🌐 Publishing

This project uses **Gradio** for the web interface. This allows for easy deployment to **Hugging Face Spaces**.
1. Create a Space on Hugging Face.
2. Upload `src/app.py` (renaming to `app.py` in root if needed) and `requirements.txt`.
3. Your model is live on the web!
