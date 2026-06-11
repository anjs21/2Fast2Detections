"""
Dataset classes, transforms, and metadata scanning utilities.
"""

import io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

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
    if csv_path is not None:
        df.to_csv(csv_path, index=False)

    print(f"Scan complete! Total images: {len(df)}")
    print("Balance matrix:")
    print(pd.crosstab(df["transform_label"], df["binary_label"]))

    return df


def _scan_root(root_path, transform_override=None):
    """Scan one root and return a clean (uncorrupted, labeled) DataFrame.

    If `transform_override` is given (e.g. "original"), the transformation label
    is set directly instead of being inferred from the path — needed for the
    original train/val folders whose paths don't encode the transform name.
    """
    root = Path(root_path)
    if not root.exists():
        print(f"  [warn] root does not exist, skipping: {root_path}")
        return pd.DataFrame()

    rows = []
    for img_path in tqdm(list(root.rglob("*")), desc=f"Scanning {root.name}"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = [p.lower() for p in img_path.relative_to(root).parts]
        binary_label, transform_label = assign_labels(parts)
        if transform_override is not None:
            transform_label = transform_override
        try:
            with Image.open(img_path) as img:
                width, height = img.size
            is_corrupted = False
        except Exception:
            width, height, is_corrupted = None, None, True
        rows.append({
            "filepath": str(img_path),
            "binary_label": binary_label,
            "transform_label": transform_label,
            "width": width,
            "height": height,
            "is_corrupted": is_corrupted,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df[(df["is_corrupted"] == False) &
                (df["binary_label"] != "unknown") &
                (df["transform_label"] != "unknown")].reset_index(drop=True)
    return df


def build_splits(original_train_dir, original_val_dir, test_subset_dir,
                 subset_per_class, seed, csv_path=None):
    """Construct leak-free train/val splits.

    - "original" images carry a curated train/val split (folders) -> honored as-is.
    - "transmitted"/"redigitalized" images live only in the test_subset tarball;
      they are split 80/20 into train/val at the IMAGE level (disjoint -> no leak),
      stratified by (binary_label, transform_label).

    Each row gets a `split` column ("train"/"val"). A balanced per-class cap is
    applied independently within each split. Returns (train_df, val_df).
    """
    print("\n--- Building leak-free splits ---")
    orig_train = _scan_root(original_train_dir, transform_override="original")
    orig_val = _scan_root(original_val_dir, transform_override="original")
    print(f"  original: {len(orig_train)} train / {len(orig_val)} val")

    other = _scan_root(test_subset_dir)
    other = other[other["transform_label"].isin(["transmitted", "redigitalized"])].reset_index(drop=True)
    print(f"  transmitted+redigitalized (test_subset): {len(other)} images")

    if len(other):
        other_train, other_val = train_test_split(
            other, test_size=0.2,
            stratify=other[["binary_label", "transform_label"]],
            random_state=seed,
        )
    else:
        other_train = other_val = other

    orig_train["split"] = "train"
    orig_val["split"] = "val"
    other_train = other_train.assign(split="train")
    other_val = other_val.assign(split="val")

    train_df = pd.concat([orig_train, other_train], ignore_index=True)
    val_df = pd.concat([orig_val, other_val], ignore_index=True)

    # Balanced per-class cap within each split
    def _balance(df):
        return pd.concat([
            grp.sample(min(len(grp), subset_per_class), random_state=seed)
            for _, grp in df.groupby(["binary_label", "transform_label"])
        ]).reset_index(drop=True)

    train_df = _balance(train_df)
    val_df = _balance(val_df)

    # Sanity: no filepath appears in both splits (true disjointness)
    overlap = set(train_df["filepath"]) & set(val_df["filepath"])
    assert not overlap, f"LEAK: {len(overlap)} files in both train and val!"

    if csv_path is not None:
        pd.concat([train_df, val_df], ignore_index=True).to_csv(csv_path, index=False)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


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
# NOTE: AI-generated-image forensics relies on low-amplitude, high-frequency
# generation/compression traces. Object-recognition augmentation (color jitter,
# rotation, RandomResizedCrop, RandomErasing) RESAMPLES and DESTROYS exactly
# those traces. We therefore (a) avoid resampling where possible by cropping
# native pixels rather than resizing, and (b) augment with the *degradations the
# task is about* — JPEG re-compression and mild sensor noise — which simulate
# the transmission/re-digitization pipeline instead of erasing its signature.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEG:
    """Randomly re-encode the PIL image as JPEG to inject compression artifacts."""

    def __init__(self, p=0.5, quality_range=(50, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        if torch.rand(1).item() >= self.p:
            return img
        q = int(torch.randint(self.quality_range[0], self.quality_range[1] + 1, (1,)).item())
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    """Add mild Gaussian noise to a tensor (post-ToTensor, pre-Normalize)."""

    def __init__(self, p=0.3, std=0.02):
        self.p = p
        self.std = std

    def __call__(self, t):
        if torch.rand(1).item() >= self.p:
            return t
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)


class ResizeIfSmaller:
    """Resize (shorter side) only if the image is smaller than `size`.

    Keeps native pixels for large images (no destructive resampling), while
    guaranteeing crops won't fail on small images.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if min(w, h) < self.size:
            return transforms.functional.resize(img, self.size)
        return img


def get_train_transform(img_size):
    """Forensics-friendly training augmentation: crop native pixels + degradations."""
    return transforms.Compose([
        ResizeIfSmaller(img_size),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        RandomJPEG(p=0.5, quality_range=(50, 95)),
        transforms.ToTensor(),
        GaussianNoise(p=0.3, std=0.02),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size):
    """Validation/test transform: aspect-preserving resize + center crop.

    Avoids the square `Resize((s, s))` distortion of the previous pipeline; only
    resizes the shorter side, then center-crops, minimizing resampling.
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # shorter side -> ~256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
