"""
Model architectures: backbone factory, MultiTaskModel, and SingleTaskModel.
"""

import torch.nn as nn
from torchvision import models


def get_backbone(name="resnet50"):
    """Return a pretrained backbone, its feature dimension, and shared_layer reference for GradNorm."""
    if name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = base.fc.in_features
        base.fc = nn.Identity()
        shared_layer = base.layer4[-1]
    elif name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = base.fc.in_features
        base.fc = nn.Identity()
        shared_layer = base.layer4[-1]
    elif name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        shared_layer = base.features[-1]
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return base, num_features, shared_layer


class MultiTaskModel(nn.Module):
    """Shared backbone with two independent classification heads."""

    def __init__(self, backbone_name="resnet50", num_binary=2, num_transform=3, dropout=0.3):
        super().__init__()
        self.backbone, num_features, self.shared_layer = get_backbone(backbone_name)

        # Task Head 1: Binary (Real vs Fake)
        self.binary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_binary),
        )

        # Task Head 2: Transformation type (Original / Transmitted / Redigitalized)
        self.transform_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_transform),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.binary_head(features), self.transform_head(features)


class SingleTaskModel(nn.Module):
    """Single-task model for unimodal baselines."""

    def __init__(self, backbone_name="resnet50", num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone, num_features, _ = get_backbone(backbone_name)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
