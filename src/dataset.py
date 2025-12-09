import os
import glob
import pandas as pd
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FisheriesDataset(Dataset):
    def __init__(self, root_dir, fold=0, n_folds=5, mode='train', transform=None, random_state=42):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Get all image paths and labels
        self.classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
        self.le = LabelEncoder()
        self.le.fit(self.classes)
        
        all_files = []
        all_labels = []
        
        if mode in ['train', 'val']:
            for cls in self.classes:
                cls_dir = os.path.join(root_dir, 'train', cls)
                files = glob.glob(os.path.join(cls_dir, '*.jpg'))
                all_files.extend(files)
                all_labels.extend([cls] * len(files))
                
            df = pd.DataFrame({'path': all_files, 'label': all_labels})
            
            # Stratified K-Fold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            
            # We just need the indices for the requested fold
            for i, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
                if i == fold:
                    if mode == 'train':
                        self.df = df.iloc[train_idx].reset_index(drop=True)
                    else:
                        self.df = df.iloc[val_idx].reset_index(drop=True)
                    break
        elif mode == 'test':
            # Assuming test data is flat or in a subfolder
            test_files = glob.glob(os.path.join(root_dir, 'test_stg1', '*.jpg')) # Update based on actual test structure
            self.df = pd.DataFrame({'path': test_files})
            
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.mode in ['train', 'val']:
            label_str = row['label']
            label = self.le.transform([label_str])[0]
            return image, label
        else:
            return image, os.path.basename(img_path)
