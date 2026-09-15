"""Model factory for the three-way comparison.

- ``resnet18_scratch``  — ResNet-18, random init: the "no transfer learning" baseline.
- ``resnet18``          — ResNet-18, ImageNet weights (optional extra comparison).
- ``efficientnet_b0``   — EfficientNet-B0, ImageNet weights, 224px.
- ``efficientnet_b3``   — EfficientNet-B3, ImageNet weights, 300px.

Each entry also names the last convolutional stage, which Grad-CAM hooks into.
"""

import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name == "resnet18_scratch":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """The final convolutional block whose activations Grad-CAM visualises."""
    if name.startswith("resnet18"):
        return model.layer4[-1]
    if name.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target defined for: {name}")
