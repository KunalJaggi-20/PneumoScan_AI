import urllib.request
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os
from PIL import Image

def download():
    url = "https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier/resolve/main/checkpoints/binary/resnet18.pt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print("Downloading alternative ResNet18 weights from Hugging Face (luanacarolina)...")
    with urllib.request.urlopen(req) as response, open('model_new.pt', 'wb') as out_file:
        out_file.write(response.read())
    print("Download complete!")

def test():
    device = torch.device("cpu")
    
    # Try loading the model
    try:
        # Check if it's state_dict or full model
        checkpoint = torch.load("model_new.pt", map_location=device)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded from dict key 'state_dict'")
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded from dict key 'model_state_dict'")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print("Loaded state_dict directly")
        else:
            model = checkpoint
            print("Loaded full model object")
            
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # ImageNet preprocessing is used by this model
    t_imagenet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Custom 0.5 preprocessing just in case
    t_huggingface = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    images = [
        ("demo_pneumonia", "static/demo_pneumonia.jpg"),
        ("demo_normal", "static/demo_normal.jpg")
    ]
    
    for name, path in images:
        img = Image.open(path).convert("RGB")
        
        # ImageNet
        prob_im = torch.sigmoid(model(t_imagenet(img).unsqueeze(0))).item()
        # HuggingFace
        prob_hf = torch.sigmoid(model(t_huggingface(img).unsqueeze(0))).item()
        
        print(f"=== {name} ===")
        print(f"  ImageNet -> Normal (0): {1.0 - prob_im:.4f}, Pneumonia (1): {prob_im:.4f}")
        print(f"  HuggingFace -> Normal (0): {1.0 - prob_hf:.4f}, Pneumonia (1): {prob_hf:.4f}\n")

if __name__ == "__main__":
    download()
    test()
