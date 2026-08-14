"""
clip_pipeline.py
----------------
Standalone semantic feature embedding pipeline based on CLIP-ConvNeXt (ConvNeXt-XXLarge).

This module mirrors the "Semantic Feature Embedding" branch of the AIDE architecture:

    Input image
        │
        ▼
    CLIP-ConvNeXt (frozen)        ← open_clip convnext_xxlarge visual trunk
        │   [B, 3072, 8, 8]
        ▼
    AdaptiveAvgPool2d(1, 1)
        │   [B, 3072]
        ▼
    Project Layer (Linear 3072→256)
        │   [B, 256]
        ▼
    Semantic feature vector

Usage
-----
    from clip_pipeline import CLIPSemanticPipeline

    pipeline = CLIPSemanticPipeline(convnext_path="/path/to/convnext_xxl.bin")
    pipeline.eval()

    # image: torch.Tensor of shape [B, 3, H, W], normalised with CLIP stats
    features = pipeline(image)   # → [B, 256]

Notes
-----
* The ConvNeXt backbone is frozen by default (requires_grad=False).
* The projection layer IS trainable — it maps 3072-d backbone output to a
  compact 256-d semantic vector suitable for downstream fusion.
* Input images should be normalised with CLIP mean/std before being passed in
  (see CLIPSemanticPipeline.clip_preprocess for a ready-made transform).
"""

import torch
import torch.nn as nn
import open_clip
from torchvision import transforms


# ---------------------------------------------------------------------------
# CLIP normalisation constants (ViT / ConvNeXt models trained by OpenAI/LAION)
# ---------------------------------------------------------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def clip_preprocess(img_size: int = 256) -> transforms.Compose:
    """
    Returns a torchvision transform that resizes and normalises an image using
    CLIP's mean and standard deviation.  Use this when your raw PIL images
    have NOT yet been normalised.

    Args:
        img_size: Side length (pixels) to resize the shorter edge to before
                  centre-cropping.  Defaults to 256 to match the ConvNeXt-XXL
                  training resolution used in the AIDE paper.

    Returns:
        A ``transforms.Compose`` object ready to be applied to a PIL image.
    """
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class CLIPSemanticPipeline(nn.Module):
    """
    Semantic feature embedding pipeline using a frozen CLIP-ConvNeXt backbone
    and a trainable linear projection head.

    Args:
        convnext_path (str | None):
            Path to the pretrained ConvNeXt-XXLarge checkpoint (.bin / .pt).
            If ``None``, the model is initialised with random weights (useful
            for testing without a checkpoint).
        proj_in_features (int):
            Channel dimension output by the ConvNeXt backbone before pooling.
            Default ``3072`` matches the ConvNeXt-XXLarge architecture.
        proj_out_features (int):
            Dimension of the output semantic feature vector.
            Default ``256`` matches the AIDE architecture.
        freeze_backbone (bool):
            Whether to freeze the ConvNeXt backbone.  Default ``True``.
    """

    def __init__(
        self,
        convnext_path: str | None = None,
        proj_in_features: int = 3072,
        proj_out_features: int = 256,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. CLIP-ConvNeXt backbone (visual trunk only, no text encoder)
        # ------------------------------------------------------------------
        print("Building CLIP-ConvNeXt-XXLarge semantic backbone...")
        model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge",
            pretrained=convnext_path,
        )

        # Keep only the visual trunk; strip the classification head pooling /
        # flattening layers so we get spatial feature maps out.
        self.backbone = model.visual.trunk
        self.backbone.head.global_pool = nn.Identity()
        self.backbone.head.flatten = nn.Identity()

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. Spatial pooling: [B, 3072, H, W] → [B, 3072]
        # ------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------------------------------
        # 3. Trainable projection layer: [B, 3072] → [B, proj_out_features]
        # ------------------------------------------------------------------
        self.proj = nn.Linear(proj_in_features, proj_out_features)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the semantic feature vector for a batch of images.

        Args:
            x: Tensor of shape ``[B, 3, H, W]`` normalised with CLIP mean/std.

        Returns:
            Tensor of shape ``[B, proj_out_features]`` — the semantic feature
            vector for each image in the batch.
        """
        # ---- Backbone (frozen) ----
        with torch.no_grad():
            feats = self.backbone(x)        # [B, 3072, H', W']

        # ---- Spatial pooling ----
        feats = self.avgpool(feats)         # [B, 3072, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B, 3072]

        # ---- Projection (trainable) ----
        semantic_vec = self.proj(feats)     # [B, proj_out_features]

        return semantic_vec

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def clip_preprocess(img_size: int = 256) -> transforms.Compose:
        """Alias for the module-level :func:`clip_preprocess` helper."""
        return clip_preprocess(img_size)

    def freeze_backbone(self) -> None:
        """Freeze all ConvNeXt backbone parameters."""
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all ConvNeXt backbone parameters for fine-tuning."""
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_backbone_output_dim(self) -> int:
        """Return the channel dimension produced by the backbone before pooling."""
        return self.proj.in_features

    def get_semantic_dim(self) -> int:
        """Return the dimension of the output semantic feature vector."""
        return self.proj.out_features


# ---------------------------------------------------------------------------
# Quick smoke-test (run as a script: python clip_pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLIPSemanticPipeline smoke-test")
    parser.add_argument(
        "--convnext_path",
        type=str,
        default=None,
        help="Path to pretrained ConvNeXt-XXLarge checkpoint (optional).",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--img_size",   type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    pipeline = CLIPSemanticPipeline(convnext_path=args.convnext_path).to(device)
    pipeline.eval()

    # Dummy input — CLIP-normalised random image batch
    dummy = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        out = pipeline(dummy)

    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")   # Expected: [B, 256]
    assert out.shape == (args.batch_size, 256), "Unexpected output shape!"
    print("Smoke-test passed ✓")
