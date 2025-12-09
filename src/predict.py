import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import FisheriesDataset
from model import FisheriesModel

def predict(args):
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Transforms
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Dataset and Loader
    # Note: FisheriesDataset needs to handle 'test' mode correctly with simple file listing
    test_dataset = FisheriesDataset(
        root_dir=args.data_dir, 
        mode='test', 
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = FisheriesModel(num_classes=8, pretrained=False)
    # Load weights
    weights_path = f"best_model_fold{args.fold}.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Weights file {weights_path} not found. Train the model first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_probs = []
    all_filenames = []
    
    print(f"Generating predictions for {len(test_dataset)} images...")
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_filenames.extend(filenames)
            
    # Concatenate all probabilities
    import numpy as np
    all_probs = np.concatenate(all_probs, axis=0)
    
    # create submission dataframe
    classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    
    # Ensure columns are in correct order as per sample submission (usually alphabetical, but need to match training class order)
    # Our dataset class sorted them: ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    # If sample submission matches this, we are good.
    
    df = pd.DataFrame(all_probs, columns=classes)
    df.insert(0, 'image', all_filenames)
    
    output_file = 'submission.csv'
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='fisheries_monitoring/data')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
     # Adjust path if running from root
    if not os.path.exists(args.data_dir):
        args.data_dir = os.path.join(os.path.dirname(__file__), '../data')

    predict(args)
