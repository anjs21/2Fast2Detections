"""
Model architectures: backbone factory, MultiTaskModel, and SingleTaskModel.

DRCT-ConvB: the multi-task model uses a ConvNeXt-Base shared backbone and, when
DRCT is enabled, an extra normalized projection head that feeds the supervised
contrastive loss (data.py mines diffusion-reconstructed "hard fakes" that the
contrastive objective separates from real images).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_backbone(name="convnext_base"):
    """Return a pretrained backbone and its feature dimension."""
    if name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = base.fc.in_features
        base.fc = nn.Identity()
    elif name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = base.fc.in_features
        base.fc = nn.Identity()
    elif name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
    elif name == "convnext_tiny":
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        num_features = base.classifier[2].in_features
        base.classifier = nn.Flatten(1)  # keep global-pooled features, drop the LayerNorm+Linear classifier
    elif name == "convnext_base":
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        num_features = base.classifier[2].in_features
        base.classifier = nn.Flatten(1)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return base, num_features


def _make_head(num_features, num_classes, dropout):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(256, num_classes),
    )


class MultiTaskModel(nn.Module):
    """Shared backbone with two classification heads and a DRCT projection head."""

    def __init__(self, backbone_name="convnext_base", num_binary=2, num_transform=3,
                 dropout=0.3, proj_dim=128, use_projection=True):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        self.use_projection = use_projection

        # Task Head 1: Binary (Real vs Fake)
        self.binary_head = _make_head(num_features, num_binary, dropout)
        # Task Head 2: Transformation type (Original / Transmitted / Redigitalized)
        self.transform_head = _make_head(num_features, num_transform, dropout)

        # DRCT contrastive projection head (normalized embedding for SupCon)
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, proj_dim),
            )

    def forward(self, x, return_proj=False):
        features = self.backbone(x)
        out_bin = self.binary_head(features)
        out_trans = self.transform_head(features)
        if return_proj and self.use_projection:
            proj = F.normalize(self.projection(features), dim=1)
            return out_bin, out_trans, proj
        return out_bin, out_trans


class SingleTaskModel(nn.Module):
    """Single-task model for unimodal baselines."""

    def __init__(self, backbone_name="convnext_base", num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        self.head = _make_head(num_features, num_classes, dropout)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
