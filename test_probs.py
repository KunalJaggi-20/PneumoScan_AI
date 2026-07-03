import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

def main():
    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model/pneumonia_model.pth", map_location=device))
    model.eval()
    
    t_imagenet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    t_huggingface = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Try no normalization
    t_none = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    images = [
        ("demo_pneumonia", "static/demo_pneumonia.jpg"),
        ("demo_normal", "static/demo_normal.jpg")
    ]
    
    for name, path in images:
        img = Image.open(path).convert("RGB")
        
        # ImageNet preprocessing
        out_im = torch.softmax(model(t_imagenet(img).unsqueeze(0)), dim=1)
        # HuggingFace preprocessing
        out_hf = torch.softmax(model(t_huggingface(img).unsqueeze(0)), dim=1)
        # No normalization
        out_no = torch.softmax(model(t_none(img).unsqueeze(0)), dim=1)
        
        print(f"=== {name} ===")
        print(f"  ImageNet -> Normal: {out_im[0][0].item():.4f}, Pneumonia: {out_im[0][1].item():.4f}")
        print(f"  HuggingFace -> Normal: {out_hf[0][0].item():.4f}, Pneumonia: {out_hf[0][1].item():.4f}")
        print(f"  No Normalization -> Normal: {out_no[0][0].item():.4f}, Pneumonia: {out_no[0][1].item():.4f}\n")

if __name__ == "__main__":
    main()
