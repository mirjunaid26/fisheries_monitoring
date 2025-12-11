import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image

# Import Models
from src.models.cnn import FisheriesResNet
from src.models.transformer import FisheriesViT
# from src.model import SimpleFishNet # Deleted

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ... (Dataset class unchanged) ...

# --- Training Function ---
def train(args):
    print(f"Using device: {DEVICE}")
    print(f"Model Type: {args.model_type}")
    
    # Data
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    dataset = FisheriesDetectionDataset(
        root_dir=args.data_dir,
        bbox_dir=args.bbox_dir,
        transform=train_transform
    )
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check data paths.")
        return
        
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Transforms
    train_ds_params = dataset
    val_ds_raw = FisheriesDetectionDataset(root_dir=args.data_dir, bbox_dir=args.bbox_dir, transform=val_transform)
    
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_ds_params, train_indices)
    val_subset = torch.utils.data.Subset(val_ds_raw, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train Size: {len(train_subset)} | Val Size: {len(val_subset)}")
    print(f"Classes: {dataset.classes}")
    
    # Model
    if args.model_type == 'cnn':
        model = FisheriesResNet(num_classes=len(dataset.classes))
    elif args.model_type == 'vit':
        model = FisheriesViT(num_classes=len(dataset.classes))
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    model = model.to(DEVICE)
    
    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    cls_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.MSELoss() 
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        run_cls = 0.0
        run_box = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels, boxes in progress:
            imgs, labels, boxes = imgs.to(DEVICE), labels.to(DEVICE), boxes.to(DEVICE)
            
            optimizer.zero_grad()
            pred_logits, pred_boxes = model(imgs)
            
            loss_c = cls_criterion(pred_logits, labels)
            loss_b = box_criterion(pred_boxes, boxes)
            
            total_loss = loss_c + 10.0 * loss_b
            total_loss.backward()
            optimizer.step()
            
            run_cls += loss_c.item()
            run_box += loss_b.item()
            
            progress.set_postfix({"C": loss_c.item(), "B": loss_b.item()})
            
        # Validation
        val_loss, val_c, val_b = validate(model, val_loader, cls_criterion, box_criterion)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train C={run_cls/len(train_loader):.3f} B={run_box/len(train_loader):.3f} | Val C={val_c:.3f} B={val_b:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, f"best_model_{args.model_type}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")

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
            
    avg_c = total_c / len(loader)
    avg_b = total_b / len(loader)
    return avg_c + 10.0 * avg_b, avg_c, avg_b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to training images")
    parser.add_argument("--bbox_dir", type=str, default="data/bounding_boxes", help="Path to bbox jsons")
    parser.add_argument("--save_dir", type=str, default="src", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "vit"], help="Model architecture")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
