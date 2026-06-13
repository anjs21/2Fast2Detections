import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CLIP_MODELS = {
    "clip_vit_l14": "openai/clip-vit-large-patch14",
    "clip_vit_b16": "openai/clip-vit-base-patch16",
}

class CLIPBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import CLIPVisionModel
        self.vision = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
        self.num_features = self.vision.config.hidden_size

    def forward(self, x):
        return self.vision(pixel_values=x).pooler_output

def get_backbone(name="resnet50"):
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
        base.classifier = nn.Flatten(1)
    elif name == "convnext_base":
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        num_features = base.classifier[2].in_features
        base.classifier = nn.Flatten(1)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return base, num_features

def apply_partial_finetune(backbone, name, trainable_stages):
    if trainable_stages is None:
        return
    for p in backbone.parameters():
        p.requires_grad = False
    if trainable_stages <= 0:
        return

    if name.startswith("resnet"):
        stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
    elif name.startswith("efficientnet") or name.startswith("convnext"):
        stages = list(backbone.features)
    elif name.startswith("clip"):
        stages = list(backbone.vision.encoder.layers)
    else:
        stages = list(backbone.children())

    for stage in stages[-trainable_stages:]:
        for p in stage.parameters():
            p.requires_grad = True

def _make_head(num_features, num_classes, hidden_dims=(256,), dropout=0.3):
    layers = []
    in_dim = num_features
    for i, h in enumerate(hidden_dims):
        layers += [
            nn.Dropout(dropout if i == 0 else dropout / 2),
            nn.Linear(in_dim, h),
            nn.ReLU(),
        ]
        in_dim = h
    layers += [
        nn.Dropout(dropout / 2 if hidden_dims else dropout),
        nn.Linear(in_dim, num_classes),
    ]
    return nn.Sequential(*layers)

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name="resnet50", num_binary=2, num_transform=3,
                 dropout=0.3, trainable_backbone_stages=None,
                 binary_head_cfg=None, transform_head_cfg=None):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        apply_partial_finetune(self.backbone, backbone_name, trainable_backbone_stages)

        bcfg = {"hidden_dims": (256,), "dropout": dropout, **(binary_head_cfg or {})}
        tcfg = {"hidden_dims": (256,), "dropout": dropout, **(transform_head_cfg or {})}
        self.binary_head = _make_head(num_features, num_binary, **bcfg)
        self.transform_head = _make_head(num_features, num_transform, **tcfg)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)
        return self.binary_head(features), self.transform_head(features)

class SingleTaskModel(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=2, dropout=0.3,
                 trainable_backbone_stages=None, head_cfg=None):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        apply_partial_finetune(self.backbone, backbone_name, trainable_backbone_stages)
        cfg = {"hidden_dims": (256,), "dropout": dropout, **(head_cfg or {})}
        self.head = _make_head(num_features, num_classes, **cfg)

    def forward(self, x):
        return self.head(self.backbone(x))

class BayarConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def _project(self):
        with torch.no_grad():
            c = self.kernel_size // 2
            w = self.weight
            w[:, :, c, c] = 0.0
            s = w.sum(dim=(2, 3), keepdim=True)
            s = torch.where(s.abs() < 1e-6, torch.ones_like(s), s)
            w /= s
            w[:, :, c, c] = -1.0

    def forward(self, x):
        self._project()
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

class DualStreamMultiTaskModel(nn.Module):
    def __init__(self, rgb_backbone_name="convnext_tiny", noise_backbone_name="resnet18",
                 num_binary=2, num_transform=3, dropout=0.3, trainable_backbone_stages=None,
                 binary_head_cfg=None, transform_head_cfg=None):
        super().__init__()
        self.rgb_backbone, d_rgb = get_backbone(rgb_backbone_name)
        apply_partial_finetune(self.rgb_backbone, rgb_backbone_name, trainable_backbone_stages)

        self.bayar = BayarConv2d(3, 3, kernel_size=5)
        self.noise_backbone, d_noise = get_backbone(noise_backbone_name)

        fused = d_rgb + d_noise
        bcfg = {"hidden_dims": (256,), "dropout": dropout, **(binary_head_cfg or {})}
        tcfg = {"hidden_dims": (256,), "dropout": dropout, **(transform_head_cfg or {})}
        self.binary_head = _make_head(fused, num_binary, **bcfg)
        self.transform_head = _make_head(fused, num_transform, **tcfg)

    def forward_features(self, x):
        f_rgb = self.rgb_backbone(x)
        f_noise = self.noise_backbone(self.bayar(x))
        return torch.cat([f_rgb, f_noise], dim=1)

    def forward(self, x):
        feat = self.forward_features(x)
        return self.binary_head(feat), self.transform_head(feat)

class DualStreamSingleTaskModel(nn.Module):
    def __init__(self, rgb_backbone_name="convnext_tiny", noise_backbone_name="resnet18",
                 num_classes=2, dropout=0.3, trainable_backbone_stages=None, head_cfg=None):
        super().__init__()
        self.rgb_backbone, d_rgb = get_backbone(rgb_backbone_name)
        apply_partial_finetune(self.rgb_backbone, rgb_backbone_name, trainable_backbone_stages)

        self.bayar = BayarConv2d(3, 3, kernel_size=5)
        self.noise_backbone, d_noise = get_backbone(noise_backbone_name)

        cfg = {"hidden_dims": (256,), "dropout": dropout, **(head_cfg or {})}
        self.head = _make_head(d_rgb + d_noise, num_classes, **cfg)

    def forward(self, x):
        f_rgb = self.rgb_backbone(x)
        f_noise = self.noise_backbone(self.bayar(x))
        return self.head(torch.cat([f_rgb, f_noise], dim=1))
