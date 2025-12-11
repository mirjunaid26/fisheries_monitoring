import torch
import torch.nn as nn
import timm

class FisheriesViT(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(FisheriesViT, self).__init__()
        # Load ViT Base
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Bounding Box Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        bbox_coords = self.regressor(features)
        return class_logits, bbox_coords
