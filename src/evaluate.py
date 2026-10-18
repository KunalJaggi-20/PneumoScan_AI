import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_dataloaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "../data"
if not os.path.exists(f"{data_dir}/train"):
    data_dir = "data"

train_loader, val_loader = get_dataloaders(data_dir)

model_path = "../model/pneumonia_model.pth"
if not os.path.exists(model_path):
    model_path = "model/pneumonia_model.pth"

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint

is_densenet = any(k.startswith("classifier") or "denseblock" in k for k in state_dict.keys())
arch = "densenet121" if is_densenet else "resnet18"
num_classes = 1

model = get_model(model_name=arch, num_classes=num_classes, pretrained=False)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        if outputs.shape[-1] == 1:
            probs = torch.sigmoid(outputs).squeeze(-1)
            preds = (probs >= 0.70).long()
        else:
            preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"], zero_division=0))