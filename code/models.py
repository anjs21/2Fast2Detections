"""
Model architectures: backbone factory, MultiTaskModel, and SingleTaskModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path


from aide import AIDE, DCT_base_Rec_Module


class AIDEBackboneWrapper(nn.Module):
    """Wrapper to prepare single-image inputs for AIDE's multi-input (DCT + image) forward pass."""

    def __init__(self, checkpoint_path=None):
        super().__init__()
        if AIDE is None:
            raise ImportError(
                "Could not import AIDE. Please ensure the AIDE repository is located at adjacent folder: '../AIDE'"
            )
        
        # Instantiate AIDE (resnet_path=None and convnext_path=None since full weights are loaded from checkpoint)
        self.aide_model = AIDE(resnet_path=None, convnext_path=None)
        
        # Replace classification head with Identity to output joint features (dim 2304)
        self.aide_model.fc = nn.Identity()
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading AIDE checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.aide_model.load_state_dict(clean_state_dict, strict=False)
        else:
            print("Warning: AIDE checkpoint path not found. Running with randomly initialized weights.")
            
        self.dct = DCT_base_Rec_Module(output=256)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Unnormalize to [0,1] for DCT
        x_unnorm = x * self.std + self.mean

        # 2. Compute DCT patches (32x32 each) — no input resizing needed
        x_minmin_list, x_maxmax_list, x_minmin1_list, x_maxmax1_list = [], [], [], []
        for i in range(B):
            x_mm, x_mx, x_mm1, x_mx1 = self.dct(x_unnorm[i])
            x_minmin_list.append(x_mm)
            x_maxmax_list.append(x_mx)
            x_minmin1_list.append(x_mm1)
            x_maxmax1_list.append(x_mx1)

        x_minmin = torch.stack(x_minmin_list)   # [B, 3, 32, 32]
        x_maxmax = torch.stack(x_maxmax_list)
        x_minmin1 = torch.stack(x_minmin1_list)
        x_maxmax1 = torch.stack(x_maxmax1_list)

        # 3. Re-normalize DCT patches for ResNet
        x_minmin  = (x_minmin  - self.mean) / self.std
        x_maxmax  = (x_maxmax  - self.mean) / self.std
        x_minmin1 = (x_minmin1 - self.mean) / self.std
        x_maxmax1 = (x_maxmax1 - self.mean) / self.std

        # 4. ResNet branch — HPF then ResNet, size-agnostic due to AdaptiveAvgPool
        aide = self.aide_model
        x_min  = aide.model_min(aide.hpf(x_minmin))
        x_max  = aide.model_max(aide.hpf(x_maxmax))
        x_min1 = aide.model_min(aide.hpf(x_minmin1))
        x_max1 = aide.model_max(aide.hpf(x_maxmax1))
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4   # [B, 2048]

        # 5. ConvNeXt branch — resize only here, only for this branch
        x_tokens = F.interpolate(x_unnorm, size=(256, 256), mode='bilinear', align_corners=False)

        with torch.no_grad():
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(x_tokens).view(1, 3, 1, 1)
            clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(x_tokens).view(1, 3, 1, 1)
            dinov2_mean = torch.tensor([0.485, 0.456, 0.406]).to(x_tokens).view(1, 3, 1, 1)
            dinov2_std  = torch.tensor([0.229, 0.224, 0.225]).to(x_tokens).view(1, 3, 1, 1)

            convnext_input = x_tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            local_feats = aide.openclip_convnext_xxl(convnext_input)   # [B, 3072, 8, 8]
            local_feats = aide.avgpool(local_feats).view(B, -1)         # [B, 3072]
            x_0 = aide.convnext_proj(local_feats)                       # [B, 256]

        # 6. Concatenate and return joint features
        features = torch.cat([x_0, x_1], dim=1)   # [B, 2304]
        return features


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
    elif name == "aide":
        checkpoint_path = None
        path = Path("/leonardo_scratch/large/userexternal/jbiebuyc/AIDE/GenImage_train.pth")
        if path.exists():
            checkpoint_path = path

        if not checkpoint_path:
            ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoint_AIDE"
            checkpoint_path = ckpt_dir / "GenImage_train.pth"
            if not checkpoint_path.exists():
                checkpoint_path = ckpt_dir / "genimage_train.pth"
            
        base = AIDEBackboneWrapper(checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None)
        num_features = 2048 + 256  # 2304 joint features (2048 from ResNet branches + 256 from ConvNeXt branch)
        shared_layer = base.aide_model.model_min.layer4[-1]
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
