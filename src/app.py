import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import os
import argparse
from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.models.cnn import FisheriesResNet
from src.models.transformer import FisheriesViT
from src.model import SimpleFishNet

# Constants
IMG_SIZE = 224
CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'] # Hardcoded for demo simplicity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def load_model(model_type, model_path):
    print(f"Loading {model_type} from {model_path}...")
    if model_type == 'cnn':
        model = FisheriesResNet(num_classes=len(CLASSES), pretrained=False)
    elif model_type == 'vit':
        model = FisheriesViT(num_classes=len(CLASSES), pretrained=False)
    else:
        model = SimpleFishNet(num_classes=len(CLASSES))
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None
        
    model.to(DEVICE)
    model.eval()
    return model

def transform_image(image):
    # Image is PIL
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # Convert to numpy
    image_np = np.array(image)
    transformed = transform(image=image_np)['image']
    return transformed.unsqueeze(0).to(DEVICE)

def predict(image, model_type):
    # Load model on the fly or keep global? For demo, we can assume a single model loaded or load based on UI
    # Let's use a global fallback for now if args passed, else just demo logic
    global MODEL
    
    input_tensor = transform_image(image)
    
    with torch.no_grad():
        class_logits, bbox_coords = MODEL(input_tensor)
        
    # Class
    probs = F.softmax(class_logits, dim=1).cpu().numpy()[0]
    top3_indices = probs.argsort()[-3:][::-1]
    confidences = {CLASSES[i]: float(probs[i]) for i in top3_indices}
    
    # BBox
    # Output is [x, y, w, h] normalized
    box = bbox_coords.cpu().numpy()[0]
    x, y, w, h = box
    
    # Draw logic
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    w_orig, h_orig = draw_img.size
    
    # Scale back
    abs_x = x * w_orig
    abs_y = y * h_orig
    abs_w = w * w_orig
    abs_h = h * h_orig
    
    # Draw rect
    draw.rectangle([abs_x, abs_y, abs_x + abs_w, abs_y + abs_h], outline="red", width=3)
    
    return draw_img, confidences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="src/best_model_cnn.pth")
    parser.add_argument("--model_type", type=str, default="cnn")
    args = parser.parse_args()
    
    global MODEL
    MODEL = load_model(args.model_type, args.model_path)
    
    if MODEL is None:
        print("Model failed to load. Initializing random model for demo purposes.")
        if args.model_type == 'cnn':
             MODEL = FisheriesResNet(num_classes=len(CLASSES), pretrained=False).to(DEVICE)
        else:
             MODEL = SimpleFishNet(num_classes=len(CLASSES)).to(DEVICE)
        MODEL.eval()

    iface = gr.Interface(
        fn=lambda img: predict(img, args.model_type),
        inputs=gr.Image(type="pil"),
        outputs=[gr.Image(type="pil", label="Detected Fish"), gr.Label(num_top_classes=3)],
        title="Fisheries Monitoring AI",
        description=f"Detecting Fish species using {args.model_type.upper()}. Upload an image.",
        examples=[] # Add examples if available
    )
    
    iface.launch(share=False)

if __name__ == "__main__":
    main()
