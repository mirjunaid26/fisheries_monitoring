import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFishNet(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleFishNet, self).__init__()
        
        # Shared Backbone - Simple 4-layer CNN
        # Input expected: (Batch, 3, 224, 224)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 28x28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14x14
        )
        
        # Adaptive pooling to ensure fixed size before fully connected layers
        # capable of handling slightly different input sizes if needed
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Flatten size = 256 * 7 * 7 = 12544
        flatten_dim = 256 * 7 * 7
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Bounding Box Regression Head
        # Outputs: x_center, y_center, width, height (normalized 0-1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
            nn.Sigmoid() # Normalize output to [0, 1] for relative coordinates
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        # Branching heads
        class_logits = self.classifier(x)
        bbox_coords = self.regressor(x)
        
        return class_logits, bbox_coords

if __name__ == "__main__":
    # Test instantiation
    model = SimpleFishNet()
    dummy_input = torch.randn(2, 3, 224, 224)
    cls, box = model(dummy_input)
    print(f"Class output shape: {cls.shape}")
    print(f"Box output shape: {box.shape}")
