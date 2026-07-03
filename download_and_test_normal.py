import urllib.request
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

def download():
    url = "https://upload.wikimedia.org/wikipedia/commons/9/9f/Chest_X-ray.jpg"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print("Downloading normal chest X-ray from Wikimedia...")
    with urllib.request.urlopen(req) as response, open('normal_test_xray.jpg', 'wb') as out_file:
        out_file.write(response.read())
    print("Download complete!")

def test():
    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model/pneumonia_model.pth", map_location=device))
    model.eval()
    
    # Preprocessing pipelines
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
    
    img = Image.open('normal_test_xray.jpg').convert('RGB')
    
    out_im = torch.softmax(model(t_imagenet(img).unsqueeze(0)), dim=1)
    out_hf = torch.softmax(model(t_huggingface(img).unsqueeze(0)), dim=1)
    
    print("\n=== Predictions on Confirmed NORMAL Chest X-Ray ===")
    print(f"  ImageNet -> Normal: {out_im[0][0].item():.4f}, Pneumonia: {out_im[0][1].item():.4f}")
    print(f"  HuggingFace -> Normal: {out_hf[0][0].item():.4f}, Pneumonia: {out_hf[0][1].item():.4f}\n")

if __name__ == "__main__":
    download()
    test()
