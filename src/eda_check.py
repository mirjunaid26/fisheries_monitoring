import os
import glob
import pandas as pd
from PIL import Image
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/train')
CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

print(f"Analyzing dataset at {os.path.abspath(DATA_DIR)}...")

stats = []

for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_dir):
        print(f"Warning: Directory {cls_dir} does not exist.")
        continue
    
    files = glob.glob(os.path.join(cls_dir, '*.jpg'))
    print(f"Class {cls}: {len(files)} images")
    
    # Check dimensions of first few images
    dims = []
    for f in files[:50]: # check first 50 to save time
        with Image.open(f) as img:
            dims.append(img.size)
    
    dim_counts = Counter(dims)
    stats.append({
        'class': cls,
        'count': len(files),
        'dims': dict(dim_counts)
    })

if not stats:
    print("No data found!")
    exit(1)

print("\nSummary:")
df = pd.DataFrame(stats)
print(df[['class', 'count']])
print("\nImage Dimensions (sample):")
for s in stats:
    print(f"{s['class']}: {s['dims']}")
