"""
Dataset classes, transforms, and metadata scanning utilities.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from config import IMAGE_EXTS, BINARY_MAP, TRANSFORM_MAP


# =============================================================================
# Metadata Scanning (SEPARATED for train/val and test)
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
# Dataset Classes
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
# Transforms / Augmentation
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
