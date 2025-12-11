import torch
import torch.onnx
from src.model import SimpleFishNet
import os

def export_to_onnx(model_path, output_path):
    print(f"Loading model from {model_path}...")
    model = SimpleFishNet(num_classes=8)
    
    # Load weights
    # map_location='cpu' is important if trained on GPU/MPS
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        verbose=True,
        input_names=['input'], 
        output_names=['class_logits', 'bbox_coords'],
        opset_version=11
    )
    print("Export complete.")

if __name__ == "__main__":
    MODEL_PATH = "src/simple_fish_net_best.pth"
    OUTPUT_PATH = "public/model.onnx"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Train the model first.")
    else:
        export_to_onnx(MODEL_PATH, OUTPUT_PATH)
