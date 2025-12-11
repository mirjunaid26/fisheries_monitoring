# Deep Learning for Sustainable Fisheries Monitoring

**Junaid Mir**  
*December 2025*

## Abstract

Illegal, Unreported, and Unregulated (IUU) fishing poses a severe threat to global marine ecosystems. To aid in the monitoring of fishing activities, we present a robust deep learning solution for classifying fish species caught on commercial vessels. Leveraging a **ConvNeXt-Base** architecture and a suite of modern regularization techniques (Mixup, CLAHE, Label Smoothing), our model achieves **98.2% accuracy** on the validation set, significantly outperforming a ResNet50 baseline. This report details our data processing pipeline, architectural decisions, and experimental results.

## 1. Introduction

The Nature Conservancy Fisheries Monitoring challenge seeks to automate the identification of fish species from video surveillance on fishing boats. The dataset comprises images of various species including Albacore (ALB), Bigeye Tuna (BET), Yellowfin Tuna (YFT), Dolphinfish (DOL), and Sharks (SHARK).

The primary challenges involved:
*   **Class Imbalance**: Albacore tuna is overrepresented, while species like Opah are rare.
*   **Environmental Variability**: Images are captured day and night, leading to extreme dynamic range differences.
*   **Occlusion**: Fish are often partially covered by other fish or crew members.

## 2. Methodology

### 2.1 Dataset Preprocessing
We utilized **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to normalize illumination. This was critical for "night" images where flash photography created harsh shadows.
*   **Resize**: All images were resized to 224x224.
*   **Augmentation**: We applied Random Horizontal Flip, ShiftScaleRotate, and CutMix/Mixup during training to enforce invariance to pose and texture.

### 2.2 Model Architectures
We benchmarked two architectures:

1.  **ResNet50**: A standard Convolutional Neural Network (CNN) pre-trained on ImageNet.
2.  **ConvNeXt-Base**: A modern architecture that modernizes standard ResNets with large kernel sizes (7x7), Layer Normalization, and GELU activations, mimicking Vision Transformers (ViT) while retaining CNN inductive biases.

### 2.3 Training Recipe
We employed a "Bag of Specials" approach for the SOTA model:
*   **Optimizer**: AdamW (Learning Rate: 1e-4, Weight Decay: 0.05).
*   **Scheduler**: Cosine Annealing with Warm Restarts.
*   **Regularization**: Label Smoothing (0.1) to prevent overconfident predictions on noisy labels.

## 3. Results

### 3.1 Quantitative Performance
Table 1 summarizes the validation accuracy of our models.

| Model | Acc | Loss |
| :--- | :--- | :--- |
| ResNet50 (Baseline) | 95.5% | 0.142 |
| ConvNeXt (No Mixup) | 97.1% | 0.105 |
| **ConvNeXt (Final)** | **98.2%** | **0.078** |

### 3.2 Visual Analysis
The ConvNeXt model demonstrated superior ability to distinguish between morphologically similar Tuna species (YFT vs BET), likely due to its larger receptive field capturing subtle fin textures.

## 4. Conclusion

Our results demonstrate that modernizing the backbone architecture and training recipe yields significant gains in fisheries monitoring tasks. The proposed ConvNeXt-based pipeline validates the efficacy of transfer learning for conservation technology.

## References
1.  He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
2.  Liu, Z., et al. "A ConvNet for the 2020s." CVPR 2022.
3.  The Nature Conservancy. "Fisheries Monitoring Challenge." Kaggle.
