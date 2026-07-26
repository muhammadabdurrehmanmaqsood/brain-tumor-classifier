import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir='/content/dataset/Training', batch_size=32, val_split=0.15, seed=42):
    # Creates and returns training and validation DataLoaders with proper augmentations.
    # 1. Define augmentations for training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Define transforms for validation (no random augmentations)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load the full dataset
    full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    # 4. Create the deterministic validation split
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # Override the validation subset's transform
    val_dataset.dataset.transform = val_transforms

    # 5. Package into DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_train_dataset.classes
