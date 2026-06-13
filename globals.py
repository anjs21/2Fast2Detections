import os
import torch
import warnings
import numpy as np

# Global Project Settings & Paths
KAGGLE_MODE = os.path.exists("/kaggle")
DATA_DIR = "/kaggle/working/data" if KAGGLE_MODE else os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
METADATA_TRAIN_VAL_CSV = "/kaggle/working/metadata_train_val.csv" if KAGGLE_MODE else os.path.join(os.path.abspath(os.path.dirname(__file__)), "metadata_train_val.csv")
RESULTS_DIR = "/kaggle/working/results" if KAGGLE_MODE else os.path.join(os.path.abspath(os.path.dirname(__file__)), "results")
CHECKPOINTS_DIR = "/kaggle/working/checkpoints" if KAGGLE_MODE else os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoints")

ORIGINAL_ROOT = os.path.join(DATA_DIR, "RRDataset_original_train_val")
ORIGINAL_TRAIN_DIR = os.path.join(ORIGINAL_ROOT, "train")
ORIGINAL_VAL_DIR = os.path.join(ORIGINAL_ROOT, "val")
TEST_SUBSET_DIR = os.path.join(DATA_DIR, "test_subset")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Hyperparameter Configuration
CONFIG = {
    "seed": int(os.environ.get("SEED", 42)),
    "img_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "subset_per_class": 1000,
    "binary_weight": 0.3,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "patience": 3,
    "grad_clip_norm": 1.0,
    "backbone": "convnext_base",
    "num_workers": 4,
    "amp": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "trainable_backbone_stages": None,
    "dual_stream": True,
    "noise_backbone": "resnet18",
    "tta_crops": 5,
    "dropout": 0.32,
    "binary_head": {"hidden_dims": [256]},
    "transform_head": {"hidden_dims": []},
}

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Set seed for reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Class label mappings
BINARY_MAP = {"real": 0, "fake": 1}
TRANSFORM_MAP = {"original": 0, "transmitted": 1, "redigitalized": 2}
INV_BINARY_MAP = {v: k for k, v in BINARY_MAP.items()}
INV_TRANSFORM_MAP = {v: k for k, v in TRANSFORM_MAP.items()}
