"""
Model architectures: backbone factory, MultiTaskModel, and SingleTaskModel.

The multi-task model is a shared backbone (ResNet50 by default) with two
classification heads — binary real/fake and 3-class transformation type.
Partial fine-tuning is supported: the early backbone stages can be frozen so only
the trailing stages and the task heads are trained (see `trainable_backbone_stages`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _make_head(num_features, num_classes, hidden_dims=(256,), dropout=0.3):
    """Build an MLP classification head.

    hidden_dims: widths of the hidden layers; () gives a single linear layer
        (a linear probe). The default (256,) reproduces the original head.
    dropout: applied before the first hidden layer, dropout/2 before each
        subsequent linear layer.

    The two task heads can be configured independently (see MultiTaskModel's
    binary_head_cfg / transform_head_cfg), e.g. a deeper/wider binary head and a
    shallow transform head.
    """
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
    """Shared backbone with two classification heads (binary + transformation)."""

    def __init__(self, backbone_name="resnet50", num_binary=2, num_transform=3,
                 dropout=0.3, trainable_backbone_stages=None,
                 binary_head_cfg=None, transform_head_cfg=None):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        apply_partial_finetune(self.backbone, backbone_name, trainable_backbone_stages)

        # Per-head config overrides the global dropout; both default to (256,)/dropout.
        bcfg = {"hidden_dims": (256,), "dropout": dropout, **(binary_head_cfg or {})}
        tcfg = {"hidden_dims": (256,), "dropout": dropout, **(transform_head_cfg or {})}
        # Task Head 1: Binary (Real vs Fake)
        self.binary_head = _make_head(num_features, num_binary, **bcfg)
        # Task Head 2: Transformation type (Original / Transmitted / Redigitalized)
        self.transform_head = _make_head(num_features, num_transform, **tcfg)

    def forward_features(self, x):
        """Pre-head feature vector (the input the two heads share)."""
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)
        out_bin = self.binary_head(features)
        out_trans = self.transform_head(features)
        return out_bin, out_trans


class SingleTaskModel(nn.Module):
    """Single-task model for unimodal baselines."""

    def __init__(self, backbone_name="resnet50", num_classes=2, dropout=0.3,
                 trainable_backbone_stages=None, head_cfg=None):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)
        apply_partial_finetune(self.backbone, backbone_name, trainable_backbone_stages)
        cfg = {"hidden_dims": (256,), "dropout": dropout, **(head_cfg or {})}
        self.head = _make_head(num_features, num_classes, **cfg)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class BayarConv2d(nn.Module):
    """Bayar & Stamm (2016) constrained high-pass convolution.

    A learnable first layer constrained to be a prediction-error / residual
    filter: each kernel's center weight is fixed to -1 and its remaining weights
    are renormalized to sum to +1 (so the kernel sums to 0). This suppresses image
    content and exposes the high-frequency noise/generation fingerprint, which is
    where the real-vs-AI signal lives. The constraint is re-projected every forward.
    """

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
            s = torch.where(s.abs() < 1e-6, torch.ones_like(s), s)  # guard div-by-~0
            w /= s                                                  # non-center weights sum to 1
            w[:, :, c, c] = -1.0                                    # -> kernel sums to 0 (high-pass)

    def forward(self, x):
        self._project()
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)


class DualStreamMultiTaskModel(nn.Module):
    """RGB stream + Bayar noise-residual stream, fused into the two task heads.

    Stream A: the usual pretrained RGB backbone (semantic + coarse cues).
    Stream B: a Bayar constrained high-pass -> small CNN on the residual, which
    captures the high-frequency generation/compression fingerprint that the RGB
    stream tends to discard (helps binary on transmitted/redigitalized images).
    The pooled features are concatenated and fed to both heads.

    `trainable_backbone_stages` controls partial fine-tuning of the RGB backbone
    only; the (small) noise backbone and the Bayar layer are always fully trained.
    """

    def __init__(self, rgb_backbone_name="convnext_tiny", noise_backbone_name="resnet18",
                 num_binary=2, num_transform=3, dropout=0.3, trainable_backbone_stages=None,
                 binary_head_cfg=None, transform_head_cfg=None):
        super().__init__()
        self.rgb_backbone, d_rgb = get_backbone(rgb_backbone_name)
        apply_partial_finetune(self.rgb_backbone, rgb_backbone_name, trainable_backbone_stages)

        self.bayar = BayarConv2d(3, 3, kernel_size=5)
        self.noise_backbone, d_noise = get_backbone(noise_backbone_name)

        fused = d_rgb + d_noise
        # Per-head config overrides the global dropout; both default to (256,)/dropout.
        bcfg = {"hidden_dims": (256,), "dropout": dropout, **(binary_head_cfg or {})}
        tcfg = {"hidden_dims": (256,), "dropout": dropout, **(transform_head_cfg or {})}
        self.binary_head = _make_head(fused, num_binary, **bcfg)
        self.transform_head = _make_head(fused, num_transform, **tcfg)

    def forward_features(self, x):
        """Fused RGB + noise-residual feature vector (the input the two heads share)."""
        f_rgb = self.rgb_backbone(x)
        f_noise = self.noise_backbone(self.bayar(x))
        return torch.cat([f_rgb, f_noise], dim=1)

    def forward(self, x):
        feat = self.forward_features(x)
        return self.binary_head(feat), self.transform_head(feat)


class DualStreamSingleTaskModel(nn.Module):
    """Single-task counterpart of DualStreamMultiTaskModel (one fused head).

    Used for the unimodal baselines when dual_stream is enabled, so the
    joint-vs-unimodal comparison holds the architecture fixed and isolates the
    effect of multi-task training alone.
    """

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
