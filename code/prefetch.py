"""
Prefetch all external resources required by the project.

This script downloads everything that would otherwise be fetched on-the-fly
during training, making subsequent runs (especially offline SLURM jobs) fully
self-contained.

Downloads:
  1. Zenodo datasets  — train/val and test tarballs (extracted into data/)
  2. Pretrained model weights — ResNet18, ResNet50, EfficientNet-B0 from torchvision
"""

import os
import sys

# ---------------------------------------------------------------------------
# Resolve project root (parent of the code/ directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))  # allow `from config import ...`

from config import download_train_val_data, download_test_data


def prefetch_datasets():
    """Download and extract the Zenodo datasets if not already present."""
    print("\n" + "=" * 60)
    print("PREFETCH: Zenodo Datasets")
    print("=" * 60)

    print("\n--- Train / Val dataset ---")
    download_train_val_data()

    print("\n--- Test dataset ---")
    download_test_data()


def prefetch_model_weights():
    """Force-download all pretrained backbone weights used by models.py.

    torchvision caches weights under $TORCH_HOME (default ~/.cache/torch/).
    Calling the weight enum's .get_state_dict() triggers the download if the
    file isn't already cached.
    """
    print("\n" + "=" * 60)
    print("PREFETCH: Pretrained Model Weights")
    print("=" * 60)

    from torchvision import models

    weights_to_fetch = [
        ("ResNet-18",        models.ResNet18_Weights.DEFAULT),
        ("ResNet-50",        models.ResNet50_Weights.DEFAULT),
        ("EfficientNet-B0",  models.EfficientNet_B0_Weights.DEFAULT),
    ]

    for name, weights in weights_to_fetch:
        print(f"\n  Downloading {name} weights ({weights}) ...")
        try:
            # This pulls the file into the torch cache if it's not there yet
            weights.get_state_dict(progress=True)
            print(f"  ✓ {name} weights ready.")
        except Exception as e:
            print(f"  ✗ {name} download failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PREFETCH — downloading all external resources")
    print("=" * 60)

    prefetch_datasets()
    prefetch_model_weights()

    print("\n" + "=" * 60)
    print("  PREFETCH COMPLETE — all resources are cached locally")
    print("=" * 60)