import torch.nn as nn
from torchvision import models

def get_resnet50_model(num_classes=4, pretrained=True):
    """
    Initializes a ResNet50 model and modifies the final fully connected layer 
    to output the specified number of classes.
    
    Args:
        num_classes (int): The number of target classes (4 for the MRI dataset).
        pretrained (bool): If True, loads ImageNet weights. If False, initializes randomly.
    """
    if pretrained:
        # Use the modern Weights enum for PyTorch
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
        
    # Extract the number of input features to the final layer
    num_ftrs = model.fc.in_features
    
    # Replace the final layer to match our specific classification task
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
