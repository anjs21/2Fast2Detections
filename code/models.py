"""
Model architectures: backbone factory, MultiTaskModel, and SingleTaskModel.

The multi-task model is a shared backbone (ResNet50 by default) with two
classification heads — binary real/fake and 3-class transformation type.
Partial fine-tuning is supported: the early backbone stages can be frozen so only
the trailing stages and the task heads are trained (see `trainable_backbone_stages`).
"""

import torch
import torch.nn as nn
from torchvision import models


# CLIP image-encoder model ids (HuggingFace). ViT-L/14 is the UniversalFakeDetect
# (Ojha et al., CVPR 2023) backbone; it expects 224x224 inputs.
CLIP_MODELS = {
    "clip_vit_l14": "openai/clip-vit-large-patch14",
    "clip_vit_b16": "openai/clip-vit-base-patch16",
}


class CLIPBackbone(nn.Module):
    """Wrap a HuggingFace CLIP vision tower to return a pooled feature vector.

    Used as a (typically frozen) backbone for the binary AI-image detector — the
    strong UniversalFakeDetect recipe. NOTE: CLIP expects CLIP-specific input
    normalization (see data.normalization_for), not ImageNet stats.
    """

    def __init__(self, model_name):
        super().__init__()
        from transformers import CLIPVisionModel
        # Force safetensors: transformers refuses .bin weights on torch < 2.6.
        self.vision = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
        self.num_features = self.vision.config.hidden_size

    def forward(self, x):
        return self.vision(pixel_values=x).pooler_output  # [B, hidden]


def get_backbone(name="resnet50"):
    """Return a pretrained backbone and its feature dimension."""
    if name in CLIP_MODELS:
        base = CLIPBackbone(CLIP_MODELS[name])
        return base, base.num_features
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


def apply_partial_finetune(backbone, name, trainable_stages):
    """Freeze the backbone except its last `trainable_stages` stages.

    - trainable_stages is None -> full fine-tuning (everything trainable, no-op).
    - trainable_stages == 0     -> frozen backbone (linear probe: only heads train).
    - trainable_stages == k     -> the last k stages stay trainable, the rest frozen.

    Stages are the four residual blocks (layer1..layer4) for ResNet, the feature
    blocks for EfficientNet/ConvNeXt. The task heads live outside the backbone and
    are always trainable.
    """
    if trainable_stages is None:
        return  # full fine-tuning

    # Freeze the whole backbone first (trainable_stages == 0 -> frozen probe).
    for p in backbone.parameters():
        p.requires_grad = False
    if trainable_stages <= 0:
        return

    # Identify the ordered list of stage modules for this backbone family, then
    # unfreeze the trailing `trainable_stages` of them.
    if name.startswith("resnet"):
        stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
    elif name.startswith("efficientnet"):
        stages = list(backbone.features)
    elif name.startswith("convnext"):
        stages = list(backbone.features)
    elif name.startswith("clip"):
        # CLIP "stages" = transformer encoder layers (last k stay trainable).
        stages = list(backbone.vision.encoder.layers)
    else:
        stages = list(backbone.children())

    for stage in stages[-trainable_stages:]:
        for p in stage.parameters():
            p.requires_grad = True


def _make_head(num_features, num_classes, dropout):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(256, num_classes),
    )


class MultiTaskModel(nn.Module):
    """Shared backbone with two classification heads (binary + transformation)."""

    def __init__(self, backbone_name="resnet50", num_binary=2, num_transform=3,
                 dropout=0.3, trainable_backbone_stages=None):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        apply_partial_finetune(self.backbone, backbone_name, trainable_backbone_stages)

        # Task Head 1: Binary (Real vs Fake)
        self.binary_head = _make_head(num_features, num_binary, dropout)
        # Task Head 2: Transformation type (Original / Transmitted / Redigitalized)
        self.transform_head = _make_head(num_features, num_transform, dropout)

    def forward(self, x):
        features = self.backbone(x)
        out_bin = self.binary_head(features)
        out_trans = self.transform_head(features)
        return out_bin, out_trans


class SingleTaskModel(nn.Module):
    """Single-task model for unimodal baselines."""

    def __init__(self, backbone_name="resnet50", num_classes=2, dropout=0.3,
                 trainable_backbone_stages=None):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        apply_partial_finetune(self.backbone, backbone_name, trainable_backbone_stages)
        self.head = _make_head(num_features, num_classes, dropout)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
