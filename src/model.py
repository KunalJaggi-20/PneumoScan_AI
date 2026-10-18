import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name="densenet121", num_classes=1, pretrained=True):
    """
    Returns the fine-tuning backbone model.
    Default: DenseNet121 with single binary output head.
    """
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return model