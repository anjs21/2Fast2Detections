"""
Computer Vision Project v2: Joint Detection of AI-Generated Images
and Post-Processing Alterations in Real-World Scenarios

Improvements over v1:
  - Fixed test data leak (separate metadata for train/val vs. test)
  - Upgraded backbone: ResNet50 (configurable)
  - Better task heads: Dropout + hidden layer
  - Richer data augmentation pipeline
  - Learning rate scheduler (CosineAnnealingLR)
  - Early stopping + best-model checkpointing
  - Gradient clipping
  - Unimodal baseline training for comparison
  - Ablation study over loss weight configurations
  - Per-transformation accuracy breakdown
  - Cross-class transformation trace analysis
  - Full metrics: F1, precision, recall, confusion matrices
  - Training curve visualization
"""

# =============================================================================
# Section 0: Imports & Device Setup
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

import os
import urllib.request
import tarfile
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time

print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

if torch.cuda.is_available():
    os.system("nvidia-smi")

# =============================================================================
# Section 1: Configuration
# =============================================================================
KAGGLE_MODE = os.path.exists("/kaggle")
DATA_DIR = "/kaggle/working/data" if KAGGLE_MODE else "./data"
METADATA_TRAIN_VAL_CSV = "/kaggle/working/metadata_train_val.csv" if KAGGLE_MODE else "./metadata_train_val.csv"
METADATA_TEST_CSV = "/kaggle/working/metadata_test.csv" if KAGGLE_MODE else "./metadata_test.csv"
RESULTS_DIR = "/kaggle/working/results" if KAGGLE_MODE else "./results"
CHECKPOINTS_DIR = "/kaggle/working/checkpoints" if KAGGLE_MODE else "./checkpoints"

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
    "backbone": "resnet18",      # Options: resnet18, resnet50, efficientnet_b0
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Silence warnings for clean console output
import warnings
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
    original_dir = os.path.join(DATA_DIR, "original")
    if os.path.exists(original_dir):
        print(f"Train/val data is already present in '{original_dir}'. Skipping download!")
        if os.path.exists(DATA_DIR):
            print("Top level folders in data directory:", os.listdir(DATA_DIR))
        return

    if not os.path.exists("train_val.tar.gz"):
        download_url(
            "https://zenodo.org/api/records/14963880/files/RRDataset_original_train_val.tar.gz/content",
            "train_val.tar.gz"
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(original_dir):
        print("Extracting train_val.tar.gz...")
        with tarfile.open("train_val.tar.gz", "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

    print("Top level folders in data directory:", os.listdir(DATA_DIR))


def download_test_data():
    """Download and selectively extract test data for missing categories.
    Optimized: skips if already extracted, or downloads to disk first to avoid slow socket streaming.
    """
    test_subset_dir = os.path.join(DATA_DIR, "test_subset")
    local_tar_path = "test_subset.tar.gz"

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


# =============================================================================
# Section 3: Metadata Scanning (SEPARATED for train/val and test)
# =============================================================================
def assign_labels(parts):
    """Assign binary and transform labels from path parts."""
    # Binary label
    if any("real" in p for p in parts):
        binary_label = "real"
    elif any("ai" in p or "fake" in p for p in parts):
        binary_label = "fake"
    else:
        binary_label = "unknown"

    # Transformation label
    if any("original" in p for p in parts):
        transform_label = "original"
    elif any(k in p for p in parts for k in ["transfer", "transmit", "internet", "social"]):
        transform_label = "transmitted"
    elif any(k in p for p in parts for k in ["redigital", "screen"]):
        transform_label = "redigitalized"
    else:
        transform_label = "unknown"

    return binary_label, transform_label


def scan_directory(root_path, csv_path):
    """Scan a directory for images and produce a metadata CSV."""
    root = Path(root_path)
    rows = []

    all_paths = list(root.rglob("*"))
    print(f"Scanning {root_path} ({len(all_paths)} items)...")

    for img_path in tqdm(all_paths):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        parts = [p.lower() for p in img_path.relative_to(root).parts]
        binary_label, transform_label = assign_labels(parts)

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                is_corrupted = False
        except Exception:
            width, height = None, None
            is_corrupted = True

        rows.append({
            "filepath": str(img_path),
            "binary_label": binary_label,
            "transform_label": transform_label,
            "width": width,
            "height": height,
            "is_corrupted": is_corrupted,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"Scan complete! Total images: {len(df)}")
    print("Balance matrix:")
    print(pd.crosstab(df["transform_label"], df["binary_label"]))

    return df


# =============================================================================
# Section 4: Dataset Class
# =============================================================================
class MultiTaskDataset(Dataset):
    """Dataset for joint real/fake + transformation classification."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_bin = torch.tensor(BINARY_MAP[row["binary_label"]], dtype=torch.long)
        label_trans = torch.tensor(TRANSFORM_MAP[row["transform_label"]], dtype=torch.long)

        return img, label_bin, label_trans


class SingleTaskDataset(Dataset):
    """Dataset for single-task (unimodal) baseline training."""

    def __init__(self, df, task, transform=None):
        """
        Args:
            task: 'binary' or 'transform'
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.task = task
        self.label_map = BINARY_MAP if task == "binary" else TRANSFORM_MAP

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        col = "binary_label" if self.task == "binary" else "transform_label"
        label = torch.tensor(self.label_map[row[col]], dtype=torch.long)

        return img, label


# =============================================================================
# Section 5: Transforms / Augmentation
# =============================================================================
def get_train_transform(img_size):
    """Strong training augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])


def get_val_transform(img_size):
    """Standard validation/test transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# Section 6: Model Architecture
# =============================================================================
def get_backbone(name="resnet50"):
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
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return base, num_features


class MultiTaskModel(nn.Module):
    """Shared backbone with two independent classification heads."""

    def __init__(self, backbone_name="resnet50", num_binary=2, num_transform=3, dropout=0.3):
        super().__init__()
        self.backbone, num_features = get_backbone(backbone_name)

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
        self.backbone, num_features = get_backbone(backbone_name)
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


# =============================================================================
# Section 7: Training Utilities
# =============================================================================
class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


class TrainingLogger:
    """Track metrics across epochs for visualization."""

    def __init__(self):
        self.history = defaultdict(list)

    def log(self, epoch, **kwargs):
        self.history["epoch"].append(epoch)
        for k, v in kwargs.items():
            self.history[k].append(v)

    def to_dataframe(self):
        return pd.DataFrame(self.history)


def train_multitask_epoch(model, loader, optimizer, criterion_bin, criterion_trans,
                          device, w1=0.5, w2=0.5, grad_clip_norm=1.0):
    """Train one epoch for the multi-task model."""
    model.train()
    running_loss, running_loss_bin, running_loss_trans = 0.0, 0.0, 0.0
    correct_bin, correct_trans, total = 0, 0, 0

    for images, labels_bin, labels_trans in loader:
        images = images.to(device)
        labels_bin = labels_bin.to(device)
        labels_trans = labels_trans.to(device)

        optimizer.zero_grad()
        out_bin, out_trans = model(images)

        loss_bin = criterion_bin(out_bin, labels_bin)
        loss_trans = criterion_trans(out_trans, labels_trans)
        total_loss = w1 * loss_bin + w2 * loss_trans

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += total_loss.item() * batch_size
        running_loss_bin += loss_bin.item() * batch_size
        running_loss_trans += loss_trans.item() * batch_size

        correct_bin += (torch.max(out_bin, 1)[1] == labels_bin).sum().item()
        correct_trans += (torch.max(out_trans, 1)[1] == labels_trans).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "loss_bin": running_loss_bin / total,
        "loss_trans": running_loss_trans / total,
        "acc_bin": correct_bin / total,
        "acc_trans": correct_trans / total,
    }


@torch.no_grad()
def evaluate_multitask(model, loader, criterion_bin, criterion_trans, device):
    """Evaluate the multi-task model and return loss + accuracy."""
    model.eval()
    running_loss_bin, running_loss_trans = 0.0, 0.0
    correct_bin, correct_trans, total = 0, 0, 0

    for images, labels_bin, labels_trans in loader:
        images = images.to(device)
        labels_bin = labels_bin.to(device)
        labels_trans = labels_trans.to(device)

        out_bin, out_trans = model(images)

        running_loss_bin += criterion_bin(out_bin, labels_bin).item() * images.size(0)
        running_loss_trans += criterion_trans(out_trans, labels_trans).item() * images.size(0)

        correct_bin += (torch.max(out_bin, 1)[1] == labels_bin).sum().item()
        correct_trans += (torch.max(out_trans, 1)[1] == labels_trans).sum().item()
        total += images.size(0)

    return {
        "loss_bin": running_loss_bin / total,
        "loss_trans": running_loss_trans / total,
        "acc_bin": correct_bin / total,
        "acc_trans": correct_trans / total,
    }


def train_singletask_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=1.0):
    """Train one epoch for a single-task model."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        correct += (torch.max(out, 1)[1] == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_singletask(model, loader, criterion, device):
    """Evaluate a single-task model."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        running_loss += criterion(out, labels).item() * images.size(0)
        correct += (torch.max(out, 1)[1] == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


# =============================================================================
# Section 8: Full Multi-Task Training Pipeline
# =============================================================================
def train_multitask_model(model, train_loader, val_loader, config, w1=0.5, w2=0.5, tag="multitask"):
    """Full training loop with LR scheduler, early stopping, and checkpointing."""
    device = config["device"]
    criterion_bin = nn.CrossEntropyLoss()
    criterion_trans = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_score = 0.0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Training [{tag}] | w1={w1}, w2={w2} | {config['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(config["epochs"]):
        # Train
        train_metrics = train_multitask_epoch(
            model, train_loader, optimizer, criterion_bin, criterion_trans,
            device, w1, w2, config["grad_clip_norm"]
        )

        # Validate
        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, device)

        # Step scheduler
        scheduler.step()

        # Combined val score for checkpointing
        val_score = (val_metrics["acc_bin"] + val_metrics["acc_trans"]) / 2

        # Log
        logger.log(
            epoch + 1,
            train_loss=train_metrics["loss"],
            train_loss_bin=train_metrics["loss_bin"],
            train_loss_trans=train_metrics["loss_trans"],
            train_acc_bin=train_metrics["acc_bin"],
            train_acc_trans=train_metrics["acc_trans"],
            val_loss_bin=val_metrics["loss_bin"],
            val_loss_trans=val_metrics["loss_trans"],
            val_acc_bin=val_metrics["acc_bin"],
            val_acc_trans=val_metrics["acc_trans"],
            lr=optimizer.param_groups[0]["lr"],
        )

        # Print progress
        print(f"\nEpoch [{epoch+1}/{config['epochs']}] "
              f"Loss: {train_metrics['loss']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  [Train] Real/Fake: {train_metrics['acc_bin']*100:.2f}% | "
              f"Transform: {train_metrics['acc_trans']*100:.2f}%")
        print(f"  [Val]   Real/Fake: {val_metrics['acc_bin']*100:.2f}% | "
              f"Transform: {val_metrics['acc_trans']*100:.2f}%")

        # Best model checkpoint
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = deepcopy(model.state_dict())
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{tag}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"  ★ New best model saved ({val_score*100:.2f}%)")

        # Early stopping
        early_stop(val_score)
        if early_stop.should_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model (val score: {best_val_score*100:.2f}%)")

    return model, logger


def train_singletask_model(model, train_loader, val_loader, config, task_name="binary"):
    """Full training loop for a single-task baseline."""
    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_acc = 0.0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Training Unimodal Baseline [{task_name}] | {config['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_singletask_epoch(
            model, train_loader, optimizer, criterion, device, config["grad_clip_norm"]
        )
        val_loss, val_acc = evaluate_singletask(model, val_loader, criterion, device)
        scheduler.step()

        logger.log(
            epoch + 1,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc,
            lr=optimizer.param_groups[0]["lr"],
        )

        print(f"\nEpoch [{epoch+1}/{config['epochs']}] "
              f"Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  [Train] Acc: {train_acc*100:.2f}%")
        print(f"  [Val]   Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_unimodal_{task_name}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"  ★ New best model saved ({val_acc*100:.2f}%)")

        early_stop(val_acc)
        if early_stop.should_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model (val acc: {best_val_acc*100:.2f}%)")

    return model, logger


# =============================================================================
# Section 9: Evaluation & Analysis Functions
# =============================================================================
@torch.no_grad()
def collect_multitask_predictions(model, loader, device):
    """Collect all predictions and labels from a multi-task model."""
    model.eval()
    all_preds_bin, all_labels_bin = [], []
    all_preds_trans, all_labels_trans = [], []
    all_probs_bin = []

    for images, labels_bin, labels_trans in loader:
        images = images.to(device)
        out_bin, out_trans = model(images)

        probs_bin = torch.softmax(out_bin, dim=1)
        all_probs_bin.extend(probs_bin.cpu().numpy())

        all_preds_bin.extend(torch.max(out_bin, 1)[1].cpu().numpy())
        all_labels_bin.extend(labels_bin.numpy())

        all_preds_trans.extend(torch.max(out_trans, 1)[1].cpu().numpy())
        all_labels_trans.extend(labels_trans.numpy())

    return {
        "preds_bin": np.array(all_preds_bin),
        "labels_bin": np.array(all_labels_bin),
        "probs_bin": np.array(all_probs_bin),
        "preds_trans": np.array(all_preds_trans),
        "labels_trans": np.array(all_labels_trans),
    }


@torch.no_grad()
def collect_singletask_predictions(model, loader, device):
    """Collect all predictions and labels from a single-task model."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        all_preds.extend(torch.max(out, 1)[1].cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def full_classification_report(preds, labels, class_names, task_title):
    """Print a full classification report with confusion matrix."""
    print(f"\n{'='*25} {task_title} {'='*25}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"{task_title} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"cm_{task_title.replace(' ', '_').replace('/', '_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    return cm


def per_transformation_breakdown(results, val_df):
    """
    Break down real/fake detection accuracy separately for each
    transformation category (original, transmitted, redigitalized).
    Also check whether the pattern differs between real and AI-generated images.
    """
    val_df_reset = val_df.reset_index(drop=True)

    preds_bin = results["preds_bin"]
    labels_bin = results["labels_bin"]
    labels_trans = results["labels_trans"]

    print(f"\n{'='*60}")
    print("PER-TRANSFORMATION REAL/FAKE ACCURACY BREAKDOWN")
    print(f"{'='*60}")

    rows = []
    for trans_id, trans_name in INV_TRANSFORM_MAP.items():
        mask = labels_trans == trans_id
        if mask.sum() == 0:
            continue

        subset_preds = preds_bin[mask]
        subset_labels = labels_bin[mask]

        # Overall accuracy for this transformation
        overall_acc = (subset_preds == subset_labels).mean()

        # Accuracy for real images under this transformation
        real_mask = subset_labels == BINARY_MAP["real"]
        real_acc = (subset_preds[real_mask] == subset_labels[real_mask]).mean() if real_mask.sum() > 0 else 0.0

        # Accuracy for fake images under this transformation
        fake_mask = subset_labels == BINARY_MAP["fake"]
        fake_acc = (subset_preds[fake_mask] == subset_labels[fake_mask]).mean() if fake_mask.sum() > 0 else 0.0

        n_total = mask.sum()
        n_real = real_mask.sum()
        n_fake = fake_mask.sum()

        print(f"\n  [{trans_name.upper()}] (n={n_total})")
        print(f"    Overall Real/Fake Acc: {overall_acc*100:.2f}%")
        print(f"    Real images correct:   {real_acc*100:.2f}% (n={n_real})")
        print(f"    Fake images correct:   {fake_acc*100:.2f}% (n={n_fake})")

        rows.append({
            "transformation": trans_name,
            "overall_acc": overall_acc,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "n_total": int(n_total),
            "n_real": int(n_real),
            "n_fake": int(n_fake),
        })

    breakdown_df = pd.DataFrame(rows)

    # --- Plot grouped bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(breakdown_df))
    width = 0.25

    bars1 = ax.bar(x - width, breakdown_df["overall_acc"] * 100, width, label="Overall", color="#4C72B0")
    bars2 = ax.bar(x, breakdown_df["real_acc"] * 100, width, label="Real images", color="#55A868")
    bars3 = ax.bar(x + width, breakdown_df["fake_acc"] * 100, width, label="Fake images", color="#C44E52")

    ax.set_xlabel("Transformation Type")
    ax.set_ylabel("Real/Fake Detection Accuracy (%)")
    ax.set_title("Real/Fake Accuracy Breakdown by Transformation Type")
    ax.set_xticks(x)
    ax.set_xticklabels(breakdown_df["transformation"].str.capitalize())
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels on bars
    ax.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=8)
    ax.bar_label(bars3, fmt='%.1f%%', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_transformation_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return breakdown_df


def cross_class_trace_analysis(results):
    """
    Investigate whether AI-generated and real images respond differently
    to the same post-processing operations.
    """
    preds_bin = results["preds_bin"]
    labels_bin = results["labels_bin"]
    preds_trans = results["preds_trans"]
    labels_trans = results["labels_trans"]
    probs_bin = results["probs_bin"]

    print(f"\n{'='*60}")
    print("CROSS-CLASS TRANSFORMATION TRACE ANALYSIS")
    print(f"{'='*60}")

    # Build a 6-cell analysis: (real/fake) x (original/transmitted/redigitalized)
    rows = []
    for bin_id, bin_name in INV_BINARY_MAP.items():
        for trans_id, trans_name in INV_TRANSFORM_MAP.items():
            mask = (labels_bin == bin_id) & (labels_trans == trans_id)
            if mask.sum() == 0:
                continue

            subset_preds_bin = preds_bin[mask]
            subset_labels_bin = labels_bin[mask]
            subset_probs = probs_bin[mask]

            bin_acc = (subset_preds_bin == subset_labels_bin).mean()
            avg_confidence = subset_probs[np.arange(len(subset_labels_bin)), subset_labels_bin].mean()
            misclass_rate = 1.0 - bin_acc

            # Transformation classification accuracy for this cell
            subset_preds_trans = preds_trans[mask]
            subset_labels_trans = labels_trans[mask]
            trans_acc = (subset_preds_trans == subset_labels_trans).mean()

            rows.append({
                "binary_class": bin_name,
                "transform_type": trans_name,
                "n_samples": int(mask.sum()),
                "binary_acc": bin_acc,
                "binary_misclass_rate": misclass_rate,
                "avg_confidence": avg_confidence,
                "transform_acc": trans_acc,
            })

    trace_df = pd.DataFrame(rows)
    print("\nDetailed 6-cell analysis (binary_class × transform_type):")
    print(trace_df.to_string(index=False, float_format="%.4f"))

    # --- Heatmap: Binary accuracy across the 2x3 grid ---
    pivot_acc = trace_df.pivot(index="binary_class", columns="transform_type", values="binary_acc")
    pivot_misclass = trace_df.pivot(index="binary_class", columns="transform_type", values="binary_misclass_rate")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(pivot_acc * 100, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0],
                vmin=50, vmax=100)
    axes[0].set_title("Real/Fake Accuracy (%) by Class × Transformation")
    axes[0].set_ylabel("True Binary Class")
    axes[0].set_xlabel("Transformation Type")

    sns.heatmap(pivot_misclass * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1],
                vmin=0, vmax=50)
    axes[1].set_title("Misclassification Rate (%) by Class × Transformation")
    axes[1].set_ylabel("True Binary Class")
    axes[1].set_xlabel("Transformation Type")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_class_traces.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # --- Confidence distribution comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (trans_id, trans_name) in enumerate(INV_TRANSFORM_MAP.items()):
        ax = axes[i]
        for bin_id, bin_name in INV_BINARY_MAP.items():
            mask = (labels_bin == bin_id) & (labels_trans == trans_id)
            if mask.sum() == 0:
                continue
            # Confidence of the correct class
            correct_conf = probs_bin[mask, bin_id]
            ax.hist(correct_conf, bins=20, alpha=0.5, label=f"{bin_name.capitalize()} images",
                    density=True, edgecolor="black", linewidth=0.5)

        ax.set_title(f"Confidence Distribution\n({trans_name.capitalize()})")
        ax.set_xlabel("P(correct class)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_distributions.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return trace_df


# =============================================================================
# Section 10: Ablation Study
# =============================================================================
def run_ablation_study(train_loader, val_loader, config):
    """
    Sweep different loss weight configurations and compare task accuracies.
    Tests whether the two tasks compete or complement each other.
    """
    weight_configs = [
        (1.0, 0.0, "binary_only"),
        (0.75, 0.25, "binary_dominant"),
        (0.5, 0.5, "equal"),
        (0.25, 0.75, "transform_dominant"),
        (0.0, 1.0, "transform_only"),
    ]

    ablation_results = []

    print(f"\n{'#'*60}")
    print("ABLATION STUDY: Loss Weight Sweep")
    print(f"{'#'*60}")

    for w1, w2, label in weight_configs:
        print(f"\n--- Config: w1={w1}, w2={w2} ({label}) ---")

        model = MultiTaskModel(backbone_name=config["backbone"]).to(config["device"])
        model, logger = train_multitask_model(
            model, train_loader, val_loader, config, w1=w1, w2=w2, tag=f"ablation_{label}"
        )

        # Final evaluation on best model
        criterion_bin = nn.CrossEntropyLoss()
        criterion_trans = nn.CrossEntropyLoss()
        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, config["device"])

        result = {
            "w1": w1,
            "w2": w2,
            "label": label,
            "val_acc_bin": val_metrics["acc_bin"],
            "val_acc_trans": val_metrics["acc_trans"],
        }
        ablation_results.append(result)
        print(f"  Final Val -> Real/Fake: {val_metrics['acc_bin']*100:.2f}%, "
              f"Transform: {val_metrics['acc_trans']*100:.2f}%")

    ablation_df = pd.DataFrame(ablation_results)

    # --- Plot Pareto frontier ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ablation_df["val_acc_bin"] * 100, ablation_df["val_acc_trans"] * 100,
            'o-', markersize=10, linewidth=2, color="#4C72B0")

    for _, row in ablation_df.iterrows():
        ax.annotate(
            f"w1={row['w1']:.2f}\nw2={row['w2']:.2f}",
            (row["val_acc_bin"] * 100, row["val_acc_trans"] * 100),
            textcoords="offset points", xytext=(10, 10), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
        )

    ax.set_xlabel("Real/Fake Validation Accuracy (%)", fontsize=12)
    ax.set_ylabel("Transformation Validation Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study: Task Accuracy Trade-off (Pareto Frontier)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_pareto.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # --- Bar chart comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ablation_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, ablation_df["val_acc_bin"] * 100, width, label="Real/Fake Acc", color="#4C72B0")
    bars2 = ax.bar(x + width/2, ablation_df["val_acc_trans"] * 100, width, label="Transform Acc", color="#DD8452")

    # Add value labels on bars
    ax.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)

    ax.set_xlabel("Weight Configuration")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Ablation Study: Effect of Loss Weights on Task Performance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w1={r['w1']}\nw2={r['w2']}" for _, r in ablation_df.iterrows()])
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_bars.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return ablation_df


# =============================================================================
# Section 11: Training Curve Visualization
# =============================================================================
def plot_training_curves(logger, title="Training Curves", filename="training_curves.png"):
    """Plot loss and accuracy curves from a TrainingLogger."""
    history = logger.to_dataframe()

    # Detect if this is a multi-task or single-task logger
    is_multitask = "train_acc_bin" in history.columns

    if is_multitask:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(history["epoch"], history["train_loss"], 'o-', label="Train Total Loss")
        axes[0, 0].plot(history["epoch"], history["train_loss_bin"], 's--', label="Train Binary Loss", alpha=0.7)
        axes[0, 0].plot(history["epoch"], history["train_loss_trans"], '^--', label="Train Transform Loss", alpha=0.7)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Val Loss
        axes[0, 1].plot(history["epoch"], history["val_loss_bin"], 's-', label="Val Binary Loss", color="orange")
        axes[0, 1].plot(history["epoch"], history["val_loss_trans"], '^-', label="Val Transform Loss", color="red")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Train Accuracy
        axes[1, 0].plot(history["epoch"], history["train_acc_bin"].apply(lambda x: x*100), 'o-', label="Binary")
        axes[1, 0].plot(history["epoch"], history["train_acc_trans"].apply(lambda x: x*100), 's-', label="Transform")
        axes[1, 0].set_title("Training Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Val Accuracy
        axes[1, 1].plot(history["epoch"], history["val_acc_bin"].apply(lambda x: x*100), 'o-', label="Binary")
        axes[1, 1].plot(history["epoch"], history["val_acc_trans"].apply(lambda x: x*100), 's-', label="Transform")
        axes[1, 1].set_title("Validation Accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(history["epoch"], history["train_loss"], 'o-', label="Train")
        axes[0].plot(history["epoch"], history["val_loss"], 's-', label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["epoch"], history["train_acc"].apply(lambda x: x*100), 'o-', label="Train")
        axes[1].plot(history["epoch"], history["val_acc"].apply(lambda x: x*100), 's-', label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Section 12: Visual Inference
# =============================================================================
def run_visual_inference(model, df, val_transform, device, num_samples=6):
    """Display model predictions on random images."""
    model.eval()
    samples = df.sample(num_samples, random_state=np.random.randint(1, 10000))

    cols = min(num_samples, 3)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(samples.iterrows()):
        raw_img = Image.open(row["filepath"]).convert("RGB")
        input_tensor = val_transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_bin, out_trans = model(input_tensor)
            p_bin = INV_BINARY_MAP[torch.max(out_bin, 1)[1].item()]
            p_trans = INV_TRANSFORM_MAP[torch.max(out_trans, 1)[1].item()]

        ax = axes[i]
        ax.imshow(raw_img)
        is_correct = (p_bin == row["binary_label"]) and (p_trans == row["transform_label"])
        title_color = "green" if is_correct else "red"
        ax.set_title(
            f"True: {row['binary_label']} | {row['transform_label']}\n"
            f"Pred: {p_bin} | {p_trans}",
            color=title_color, fontsize=11
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "visual_inference.png"), dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Section 13: Comparative Summary
# =============================================================================
def print_comparative_summary(unimodal_bin_acc, unimodal_trans_acc,
                              multitask_bin_acc, multitask_trans_acc):
    """Compare unimodal baselines against multi-task model."""
    print(f"\n{'#'*60}")
    print("UNIMODAL vs MULTI-TASK COMPARISON")
    print(f"{'#'*60}")

    data = {
        "Model": ["Unimodal (Binary only)", "Unimodal (Transform only)", "Multi-task (Joint)"],
        "Real/Fake Acc (%)": [
            f"{unimodal_bin_acc*100:.2f}",
            "N/A",
            f"{multitask_bin_acc*100:.2f}",
        ],
        "Transform Acc (%)": [
            "N/A",
            f"{unimodal_trans_acc*100:.2f}",
            f"{multitask_trans_acc*100:.2f}",
        ],
    }

    comparison_df = pd.DataFrame(data)
    print(comparison_df.to_string(index=False))

    # Determine if joint training helps
    bin_delta = multitask_bin_acc - unimodal_bin_acc
    trans_delta = multitask_trans_acc - unimodal_trans_acc

    print(f"\n  Delta (Binary):    {bin_delta*100:+.2f}% "
          f"({'↑ improved' if bin_delta > 0 else '↓ degraded' if bin_delta < 0 else '= unchanged'})")
    print(f"  Delta (Transform): {trans_delta*100:+.2f}% "
          f"({'↑ improved' if trans_delta > 0 else '↓ degraded' if trans_delta < 0 else '= unchanged'})")

    if bin_delta > 0 and trans_delta > 0:
        print("\n  ✓ Joint training IMPROVES both tasks — the tasks complement each other.")
    elif bin_delta < 0 and trans_delta < 0:
        print("\n  ✗ Joint training DEGRADES both tasks — the tasks compete for capacity.")
    else:
        print("\n  ~ Joint training has mixed effects — partial complementarity.")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Unimodal\n(Binary)", "Unimodal\n(Transform)", "Multi-task\n(Joint)"]
    bin_accs = [unimodal_bin_acc * 100, 0, multitask_bin_acc * 100]
    trans_accs = [0, unimodal_trans_acc * 100, multitask_trans_acc * 100]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, bin_accs, width, label="Real/Fake Acc", color="#4C72B0")
    bars2 = ax.bar(x + width/2, trans_accs, width, label="Transform Acc", color="#DD8452")

    # Add value labels on bars, showing only non-zero validation accuracy values
    labels1 = [f'{val:.1f}%' if val > 0 else '' for val in bin_accs]
    labels2 = [f'{val:.1f}%' if val > 0 else '' for val in trans_accs]
    ax.bar_label(bars1, labels=labels1, padding=3, fontsize=9)
    ax.bar_label(bars2, labels=labels2, padding=3, fontsize=9)

    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Unimodal Baselines vs Multi-Task Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "unimodal_vs_multitask.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return comparison_df


# =============================================================================
# Section 14: MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()

    # ------------------------------------------------------------------
    # PHASE 1: DATA PREPARATION
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 1: DATA PREPARATION")
    print("="*60)

    # 1a. Download train/val data and test data
    download_train_val_data()
    download_test_data()

    # 1b. Scan DATA_DIR for all categories
    if os.path.exists(METADATA_TRAIN_VAL_CSV):
        print(f"Loading existing metadata from {METADATA_TRAIN_VAL_CSV}")
        df_train_val = pd.read_csv(METADATA_TRAIN_VAL_CSV)
        # Check if the loaded metadata contains all 3 transformation categories
        available_classes = set(df_train_val["transform_label"].dropna().unique())
        if len(available_classes.intersection({"original", "transmitted", "redigitalized"})) < 3:
            print("Cached metadata is incomplete (missing classes). Re-scanning...")
            df_train_val = scan_directory(DATA_DIR, METADATA_TRAIN_VAL_CSV)
    else:
        df_train_val = scan_directory(DATA_DIR, METADATA_TRAIN_VAL_CSV)

    # 1c. Filter out corrupted/unknown
    df_clean = df_train_val[
        (df_train_val["is_corrupted"] == False) &
        (df_train_val["binary_label"] != "unknown") &
        (df_train_val["transform_label"] != "unknown")
    ]

    # 1d. Balanced subset
    balanced_df = (
        df_clean.groupby(["binary_label", "transform_label"], group_keys=False)
        .apply(lambda x: x.sample(min(len(x), CONFIG["subset_per_class"]),
                                   random_state=CONFIG["seed"]))
        .reset_index(drop=True)
    )

    # 1e. Stratified 80/20 split
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=0.2,
        stratify=balanced_df[["binary_label", "transform_label"]],
        random_state=CONFIG["seed"],
    )

    print(f"\nTrain subset size: {len(train_df)}")
    print(f"Validation subset size: {len(val_df)}")
    print("\nTraining balance matrix:")
    print(pd.crosstab(train_df["transform_label"], train_df["binary_label"]))

    # 1f. Hardware / subset documentation
    print(f"\n--- Hardware & Subset Documentation ---")
    print(f"Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Subset per class: {CONFIG['subset_per_class']}")
    print(f"Total training images: {len(train_df)}")
    print(f"Total validation images: {len(val_df)}")
    print(f"Backbone: {CONFIG['backbone']}")

    # 1g. Create DataLoaders
    train_tfm = get_train_transform(CONFIG["img_size"])
    val_tfm = get_val_transform(CONFIG["img_size"])

    # Multi-task dataloaders
    mt_train_dataset = MultiTaskDataset(train_df, transform=train_tfm)
    mt_val_dataset = MultiTaskDataset(val_df, transform=val_tfm)
    mt_train_loader = DataLoader(mt_train_dataset, batch_size=CONFIG["batch_size"],
                                 shuffle=True, num_workers=2, pin_memory=True)
    mt_val_loader = DataLoader(mt_val_dataset, batch_size=CONFIG["batch_size"],
                                shuffle=False, num_workers=2, pin_memory=True)

    # Single-task dataloaders (for unimodal baselines)
    st_bin_train = DataLoader(
        SingleTaskDataset(train_df, "binary", train_tfm),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    st_bin_val = DataLoader(
        SingleTaskDataset(val_df, "binary", val_tfm),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True
    )
    st_trans_train = DataLoader(
        SingleTaskDataset(train_df, "transform", train_tfm),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    st_trans_val = DataLoader(
        SingleTaskDataset(val_df, "transform", val_tfm),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True
    )

    # ------------------------------------------------------------------
    # PHASE 2: UNIMODAL BASELINES
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 2: UNIMODAL BASELINES")
    print("="*60)

    # 2a. Binary-only baseline
    model_bin_only = SingleTaskModel(
        backbone_name=CONFIG["backbone"], num_classes=2
    ).to(CONFIG["device"])
    model_bin_only, logger_bin = train_singletask_model(
        model_bin_only, st_bin_train, st_bin_val, CONFIG, task_name="binary"
    )
    plot_training_curves(logger_bin, "Unimodal Baseline: Binary (Real/Fake)",
                         "curves_unimodal_binary.png")

    # Evaluate
    preds_bin_uni, labels_bin_uni = collect_singletask_predictions(
        model_bin_only, st_bin_val, CONFIG["device"]
    )
    unimodal_bin_acc = (preds_bin_uni == labels_bin_uni).mean()
    full_classification_report(preds_bin_uni, labels_bin_uni,
                               ["Real", "Fake"], "Unimodal Binary Baseline")

    # 2b. Transform-only baseline
    model_trans_only = SingleTaskModel(
        backbone_name=CONFIG["backbone"], num_classes=3
    ).to(CONFIG["device"])
    model_trans_only, logger_trans = train_singletask_model(
        model_trans_only, st_trans_train, st_trans_val, CONFIG, task_name="transform"
    )
    plot_training_curves(logger_trans, "Unimodal Baseline: Transformation Type",
                         "curves_unimodal_transform.png")

    # Evaluate
    preds_trans_uni, labels_trans_uni = collect_singletask_predictions(
        model_trans_only, st_trans_val, CONFIG["device"]
    )
    unimodal_trans_acc = (preds_trans_uni == labels_trans_uni).mean()
    full_classification_report(preds_trans_uni, labels_trans_uni,
                               ["Original", "Transmitted", "Redigitalized"],
                               "Unimodal Transform Baseline")

    # ------------------------------------------------------------------
    # PHASE 3: MULTI-TASK JOINT TRAINING
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 3: MULTI-TASK JOINT TRAINING")
    print("="*60)

    model_mt = MultiTaskModel(backbone_name=CONFIG["backbone"]).to(CONFIG["device"])
    model_mt, logger_mt = train_multitask_model(
        model_mt, mt_train_loader, mt_val_loader, CONFIG, w1=0.5, w2=0.5, tag="multitask_equal"
    )
    plot_training_curves(logger_mt, "Multi-Task Joint Training (w1=0.5, w2=0.5)",
                         "curves_multitask.png")

    # Full evaluation
    results = collect_multitask_predictions(model_mt, mt_val_loader, CONFIG["device"])

    full_classification_report(results["preds_bin"], results["labels_bin"],
                               ["Real", "Fake"], "Multi-Task Real/Fake")
    full_classification_report(results["preds_trans"], results["labels_trans"],
                               ["Original", "Transmitted", "Redigitalized"],
                               "Multi-Task Transformation")

    multitask_bin_acc = (results["preds_bin"] == results["labels_bin"]).mean()
    multitask_trans_acc = (results["preds_trans"] == results["labels_trans"]).mean()

    # ------------------------------------------------------------------
    # PHASE 4: UNIMODAL vs MULTI-TASK COMPARISON
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 4: UNIMODAL vs MULTI-TASK COMPARISON")
    print("="*60)

    comparison_df = print_comparative_summary(
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc
    )

    # ------------------------------------------------------------------
    # PHASE 5: PER-TRANSFORMATION ACCURACY BREAKDOWN
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 5: PER-TRANSFORMATION ACCURACY BREAKDOWN")
    print("="*60)

    breakdown_df = per_transformation_breakdown(results, val_df)

    # ------------------------------------------------------------------
    # PHASE 6: CROSS-CLASS TRANSFORMATION TRACE ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 6: CROSS-CLASS TRANSFORMATION TRACE ANALYSIS")
    print("="*60)

    trace_df = cross_class_trace_analysis(results)

    # ------------------------------------------------------------------
    # PHASE 7: ABLATION STUDY
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 7: ABLATION STUDY (Loss Weight Sweep)")
    print("="*60)

    ablation_df = run_ablation_study(mt_train_loader, mt_val_loader, CONFIG)

    # ------------------------------------------------------------------
    # PHASE 8: VISUAL INFERENCE
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 8: VISUAL INFERENCE")
    print("="*60)

    run_visual_inference(model_mt, val_df, val_tfm, CONFIG["device"], num_samples=6)

    # ------------------------------------------------------------------
    # PHASE 9: SAVE FINAL RESULTS
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 9: SAVING RESULTS")
    print("="*60)

    # Save all results to a JSON summary
    final_summary = {
        "config": {k: str(v) for k, v in CONFIG.items()},
        "unimodal_binary_acc": float(unimodal_bin_acc),
        "unimodal_transform_acc": float(unimodal_trans_acc),
        "multitask_binary_acc": float(multitask_bin_acc),
        "multitask_transform_acc": float(multitask_trans_acc),
        "ablation_results": ablation_df.to_dict(orient="records"),
        "per_transformation_breakdown": breakdown_df.to_dict(orient="records"),
        "cross_class_traces": trace_df.to_dict(orient="records"),
        "total_training_time_minutes": (time.time() - start_time) / 60,
    }

    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    # Save the final multi-task model
    final_model_path = os.path.join(CHECKPOINTS_DIR, "final_multitask_model.pth")
    torch.save(model_mt.state_dict(), final_model_path)

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE! Total time: {elapsed:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Model checkpoints: {CHECKPOINTS_DIR}/")
    print(f"Experiment summary: {summary_path}")
    print(f"{'='*60}")
