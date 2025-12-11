import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.simple_cnn.model import SimpleFishNet

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# --- Dataset ---
class FisheriesDetectionDataset(Dataset):
    def __init__(self, root_dir, bbox_dir, transform=None):
        """
        Args:
            root_dir: Path to image directory (e.g., 'data/raw/train')
            bbox_dir: Path to directory containing JSONs (e.g., 'data/bounding_boxes')
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
        
        self.bbox_map = self._load_bboxes(bbox_dir)

    def _load_samples(self):
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def _load_bboxes(self, bbox_dir):
        bbox_map = {}
        # Ensure we look for files that exist
        if not os.path.exists(bbox_dir):
            print(f"Warning: BBox directory {bbox_dir} not found. Training without boxes.")
            return bbox_map

        for fname in os.listdir(bbox_dir):
            if fname.endswith('.json'):
                path = os.path.join(bbox_dir, fname)
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Data is a list of dicts: [{'filename': '...', 'annotations': [...]}, ...]
                    for entry in data:
                        filename = entry.get('filename', '')
                        # Some JSONs might have directory prefixes, e.g. "path/img_123.jpg"
                        # We only care about the basename "img_123.jpg"
                        basename = os.path.basename(filename)
                        
                        annotations = entry.get('annotations', [])
                        if annotations:
                            # Use the first annotation
                            rect = annotations[0]
                            # Store raw rect: x, y, w, h
                            bbox_map[basename] = [rect['x'], rect['y'], rect['width'], rect['height']]
                        else:
                            # Explicitly no box
                            bbox_map[basename] = [0, 0, 0, 0]
        return bbox_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        basename = os.path.basename(img_path)
        
        # Load Image
        image = Image.open(img_path).convert('RGB')
        w_orig, h_orig = image.size
        
        # Get BBox (if valid)
        # Default to 0,0,0,0 (representing no box or background)
        # Format: [x, y, w, h]
        raw_box = self.bbox_map.get(basename, [0, 0, 0, 0])
        
        # Transform
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        # Normalize BBox to [0, 1] relative to ORIGINAL image size
        # If box is [0,0,0,0] it remains 0s
        x, y, w, h = raw_box
        norm_box = torch.tensor([
            x / w_orig,
            y / h_orig,
            w / w_orig,
            h / h_orig
        ], dtype=torch.float32)
        
        # Clip to ensure 0-1 range
        norm_box = torch.clamp(norm_box, 0.0, 1.0)

        return image_tensor, torch.tensor(label, dtype=torch.long), norm_box

# --- Training Function ---
def train(args):
    print(f"Using device: {DEVICE}")
    
    # Transforms (Resize only, no complex augs for simplicity first)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Use standard ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data
    dataset = FisheriesDetectionDataset(
        root_dir=args.data_dir,
        bbox_dir=args.bbox_dir,
        transform=train_transform
    )
    
    # Simple split (using subset of training for valid to keep it simple if valid is not sep)
    # Ideally use stratified split, but for this "simple" experiment, standard random split is ok
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset Size: {len(dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    # Model
    model = SimpleFishNet(num_classes=len(dataset.classes))
    model = model.to(DEVICE)
    
    # Optimization
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    cls_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.MSELoss() # Simple regression loss
    
    # Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_cls_loss = 0.0
        running_box_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels, boxes in progress:
            imgs, labels, boxes = imgs.to(DEVICE), labels.to(DEVICE), boxes.to(DEVICE)
            
            optimizer.zero_grad()
            pred_logits, pred_boxes = model(imgs)
            
            # Loss Calculation
            # 1. Classification
            loss_c = cls_criterion(pred_logits, labels)
            
            # 2. Regression
            # Only compute box loss for images that ACTUALLY have boxes?
            # For simplicity, we'll assume the model learns to output 0s for missing boxes
            # Or we can mask it. Let's do a simple mask: if box sum > 0
            loss_b = box_criterion(pred_boxes, boxes)
            
            # Weighted Sum (Alpha=10 to bring MSE up to CE scale usually)
            total_loss = loss_c + 10.0 * loss_b
            
            total_loss.backward()
            optimizer.step()
            
            running_cls_loss += loss_c.item()
            running_box_loss += loss_b.item()
            
            progress.set_postfix({
                "Cls": loss_c.item(),
                "Box": loss_b.item()
            })
            
        # Validation
        val_cls_loss, val_box_loss = validate(model, val_loader, cls_criterion, box_criterion)
        
        # Save Checkpoint
        val_total = val_cls_loss + 10.0 * val_box_loss
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(model.state_dict(), os.path.join(args.save_dir, "simple_fish_net_best.pth"))
            print(f"Saved Best Model (Loss: {val_total:.4f})")
            
        print(f"Epoch {epoch+1} Summary: Train Cls={running_cls_loss/len(train_loader):.4f} Box={running_box_loss/len(train_loader):.4f} | Val Cls={val_cls_loss:.4f} Box={val_box_loss:.4f}")

def validate(model, loader, cls_crit, box_crit):
    model.eval()
    total_c = 0.0
    total_b = 0.0
    with torch.no_grad():
        for imgs, labels, boxes in loader:
            imgs, labels, boxes = imgs.to(DEVICE), labels.to(DEVICE), boxes.to(DEVICE)
            pred_logits, pred_boxes = model(imgs)
            total_c += cls_crit(pred_logits, labels).item()
            total_b += box_crit(pred_boxes, boxes).item()
    return total_c / len(loader), total_b / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to training images")
    parser.add_argument("--bbox_dir", type=str, default="data/bounding_boxes", help="Path to bbox jsons")
    parser.add_argument("--save_dir", type=str, default="src/simple_cnn", help="Where to save checkpoints")
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
