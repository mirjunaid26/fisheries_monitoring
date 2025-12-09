import torch
import torch.nn as nn
from torchvision import models

class FisheriesModel(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(FisheriesModel, self).__init__()
        # Use new weights API or old one? Let's use old 'pretrained=True' for simplicity or 'weights' if updated.
        # Check pytorch version installed. 2.5.1 supports 'weights' but pretrained=True often deprecated.
        # Safe call:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
