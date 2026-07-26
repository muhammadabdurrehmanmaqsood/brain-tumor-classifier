import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

# Import our custom modules
from dataset import get_data_loaders
from model import get_resnet50_model

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    # Executes the training and validation loop, saving the best model weights.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it has the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"*** New best model found with accuracy: {best_acc:.4f} ***")

        print()

    print(f'Training complete. Best Validation Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description='Train a ResNet50 model for tumor classification.')
    parser.add_argument('--data_dir', type=str, default='/content/dataset/Training', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='resnet50_best.pth', help='Path to save the best model weights')
    
    args = parser.parse_args()

    # 2. Hardware Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Data
    print("Loading data...")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=args.data_dir, 
        batch_size=args.batch_size
    )
    dataloaders = {'train': train_loader, 'val': val_loader}
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 4. Initialize Model
    print("Initializing model...")
    model = get_resnet50_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # 5. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. Train the Model
    print("Starting training...")
    best_model = train_model(model, dataloaders, criterion, optimizer, args.epochs, device)

    # 7. Save the Best Weights
    torch.save(best_model.state_dict(), args.save_path)
    print(f"Model saved successfully to {args.save_path}")

if __name__ == '__main__':
    main()
