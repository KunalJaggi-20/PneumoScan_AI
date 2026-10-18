import os
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from model import get_model

# ------------------------
# 1. Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "../data"
if not os.path.exists(f"{data_dir}/train"):
    data_dir = "data"

train_loader, val_loader = get_dataloaders(data_dir)

model = get_model(model_name="densenet121", num_classes=1, pretrained=True)
model = model.to(device)

head_params = model.classifier.parameters() if hasattr(model, "classifier") else model.fc.parameters()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(head_params, lr=0.0001)

# ------------------------
# 2. Training config
# ------------------------
num_epochs = 5
best_val_acc = 0.0

model_dir = "../model" if os.path.exists("../model") else "model"
os.makedirs(model_dir, exist_ok=True)
save_path = os.path.join(model_dir, "pneumonia_model.pth")

# ------------------------
# 3. Training loop
# ------------------------
for epoch in range(num_epochs):

    # -------- TRAIN --------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = (torch.sigmoid(outputs) >= 0.70).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / max(total, 1)

    # -------- VALIDATION --------
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.70).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / max(val_total, 1)

    # -------- LOGGING --------
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # -------- SAVE BEST MODEL --------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print("Best model saved!")

print("\nTraining complete!")