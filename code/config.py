"""
Configuration, constants, and data download utilities.
"""

import os
import torch
import urllib.request
import tarfile
import shutil
import numpy as np
import warnings
from tqdm import tqdm


# =============================================================================
# Device Setup
# =============================================================================
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

if torch.cuda.is_available():
    os.system("nvidia-smi")

# =============================================================================
# Paths & Configuration
# =============================================================================
# PROJECT_ROOT is the parent of the code/ directory (i.e. the cv-project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

KAGGLE_MODE = os.path.exists("/kaggle")
DATA_DIR = "/kaggle/working/data" if KAGGLE_MODE else os.path.join(PROJECT_ROOT, "data")
METADATA_TRAIN_VAL_CSV = "/kaggle/working/metadata_train_val.csv" if KAGGLE_MODE else os.path.join(PROJECT_ROOT, "metadata_train_val.csv")
METADATA_TEST_CSV = "/kaggle/working/metadata_test.csv" if KAGGLE_MODE else os.path.join(PROJECT_ROOT, "metadata_test.csv")
RESULTS_DIR = "/kaggle/working/results" if KAGGLE_MODE else os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = "/kaggle/working/checkpoints" if KAGGLE_MODE else os.path.join(PROJECT_ROOT, "checkpoints")

# Source roots for the leak-free split (see data.build_splits):
#   - "original" images come WITH a predefined train/val split (honored as-is)
#   - "transmitted"/"redigitalized" images live only in the test_subset tarball
#     and are split 80/20 into train/val at the image level (disjoint -> no leak)
ORIGINAL_ROOT = os.path.join(DATA_DIR, "RRDataset_original_train_val")
ORIGINAL_TRAIN_DIR = os.path.join(ORIGINAL_ROOT, "train")
ORIGINAL_VAL_DIR = os.path.join(ORIGINAL_ROOT, "val")
TEST_SUBSET_DIR = os.path.join(DATA_DIR, "test_subset")

# DRCT diffusion-reconstructed "hard fake" images are written here (offline step)
DRCT_RECON_DIR = os.path.join(DATA_DIR, "drct_recon")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

CONFIG = {
    "seed": 42,
    "img_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "subset_per_class": 1000,
    "patience": 3,               # Early stopping patience
    "grad_clip_norm": 1.0,       # Gradient clipping max norm
    "backbone": "convnext_base", # DRCT-ConvB backbone. Options: convnext_base, convnext_tiny, resnet50, resnet18, efficientnet_b0
    "num_workers": 4,
    "amp": True,                 # Mixed-precision training (no-op on CPU)
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ---- DRCT (Diffusion Reconstruction Contrastive Training) ----
    "use_drct": True,            # Enable supervised-contrastive head + reconstructed hard-fakes
    "contrastive_weight": 0.2,   # Lambda on the SupCon loss added to the joint CE loss
    "contrastive_temp": 0.1,     # SupCon temperature
    "drct_reconstruct": True,    # Run the offline SD reconstruction step (needs diffusers + GPU)
    "drct_sd_model": "stabilityai/stable-diffusion-2-1-base",
    "drct_recon_strength": 0.2,  # img2img noising strength for reconstructions (low = near-copy)
    "drct_recon_per_class": 500, # How many real train images to reconstruct as hard-fakes
}

# Silence warnings for clean console output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Label mappings
BINARY_MAP = {"real": 0, "fake": 1}
TRANSFORM_MAP = {"original": 0, "transmitted": 1, "redigitalized": 2}
INV_BINARY_MAP = {v: k for k, v in BINARY_MAP.items()}
INV_TRANSFORM_MAP = {v: k for k, v in TRANSFORM_MAP.items()}


# =============================================================================
# Download Utilities
# =============================================================================
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"Downloading {output_path}") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_train_val_data():
    """Download and extract the train/val dataset."""
    original_dir = os.path.join(DATA_DIR, "RRDataset_original_train_val")
    if os.path.exists(original_dir):
        print(f"Train/val data is already present in '{original_dir}'. Skipping download!")
        if os.path.exists(DATA_DIR):
            print("Top level folders in data directory:", os.listdir(DATA_DIR))
        return

    tar_path = os.path.join(PROJECT_ROOT, "train_val.tar.gz")
    if not os.path.exists(tar_path):
        download_url(
            "https://zenodo.org/api/records/14963880/files/RRDataset_original_train_val.tar.gz/content",
            tar_path
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(original_dir):
        print("Extracting train_val.tar.gz...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

    print("Top level folders in data directory:", os.listdir(DATA_DIR))


def download_test_data():
    """Download and selectively extract test data for missing categories.
    Optimized: skips if already extracted, or downloads to disk first to avoid slow socket streaming.
    """
    test_subset_dir = os.path.join(DATA_DIR, "test_subset")
    local_tar_path = os.path.join(PROJECT_ROOT, "test_subset.tar.gz")

    # 1. Quick Check: Are the files already extracted from a previous run?
    if os.path.exists(test_subset_dir):
        image_count = 0
        for root, dirs, files in os.walk(test_subset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_count += 1
        if image_count >= 3800:
            print(f"Test subset is already extracted ({image_count} images found). Skipping download!")
            return

    # 2. Local Tar Check: Do we have the tarball downloaded on disk?
    if not os.path.exists(local_tar_path):
        url = "https://zenodo.org/api/records/14963880/files/RRDataset_test.tar.gz/content"
        download_url(url, local_tar_path)

    # 3. Clean and Extract locally (extremely fast compared to streaming over network socket)
    shutil.rmtree(test_subset_dir, ignore_errors=True)
    os.makedirs(test_subset_dir, exist_ok=True)

    counts = {"redigital_real": 0, "transmitted_real": 0, "transmitted_ai": 0}
    count_redigital_ai = 0
    LIMIT = 1000

    print("Extracting target categories locally from test_subset.tar.gz...")
    with tarfile.open(local_tar_path, "r:gz") as tar:
        for member in tqdm(tar):
            if not (member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg'))):
                continue
            parts = member.name.lower().split('/')
            is_real = "real" in parts
            is_ai = "ai" in parts or "fake" in parts
            is_redigital = "redigital" in parts
            is_transmitted = any(k in parts for k in ["transfer", "transmitted", "internet"])

            # Pass 1: redigital_real, transmitted_real, transmitted_ai
            key = None
            if is_redigital and is_real:
                key = "redigital_real"
            elif is_transmitted and is_real:
                key = "transmitted_real"
            elif is_transmitted and is_ai:
                key = "transmitted_ai"

            if key and counts[key] < LIMIT:
                tar.extract(member, path=test_subset_dir)
                counts[key] += 1

            # Pass 2: redigital_ai
            if is_redigital and is_ai and count_redigital_ai < LIMIT:
                tar.extract(member, path=test_subset_dir)
                count_redigital_ai += 1

            # Stop early if all targets are met
            if all(c >= LIMIT for c in counts.values()) and count_redigital_ai >= LIMIT:
                break

    print(f"Extraction complete! Final counts:")
    print(f"  - redigital_real: {counts['redigital_real']}")
    print(f"  - transmitted_real: {counts['transmitted_real']}")
    print(f"  - transmitted_ai: {counts['transmitted_ai']}")
    print(f"  - redigital_ai: {count_redigital_ai}")
