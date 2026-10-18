from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def get_dataloaders(data_dir="data", batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_base = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_base = datasets.ImageFolder(root=f"{data_dir}/train", transform=val_transform)

    train_size = int(0.8 * len(train_base))
    indices = list(range(len(train_base)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader