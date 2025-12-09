# Kaggle Fisheries Monitoring Challenge

This repository contains the solution for [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/competitions/the-nature-conservancy-fisheries-monitoring) Kaggle competition.

## Structure
- `data/`: Dataset (ignored in git)
- `src/`: Source code
    - `dataset.py`: Data loading and augmentation
    - `model.py`: ResNet50 model definition
    - `train.py`: Training loop
    - `predict.py`: Inference script

## Usage

### 1. Environment
Create the conda environment:
```bash
conda create -n fisheries python=3.10 pandas matplotlib seaborn scikit-learn pytorch torchvision torchaudio albumentations -c pytorch -c conda-forge
conda activate fisheries
```

### 2. Training
Train the model (e.g., for 10 epochs):
```bash
python src/train.py --epochs 10
```

### 3. Inference
Generate submission file:
```bash
python src/predict.py --data_dir data
```
The output file `src/submission.csv` will be generated.

## Results
- **Model**: ResNet50 (Pretrained)
- **Validation Accuracy**: ~95.5% (Fold 0, Epoch 10)
- **Loss**: CrossEntropy
