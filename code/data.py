"""
Dataset classes, transforms, and metadata scanning utilities.
"""

import io
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from config import IMAGE_EXTS, BINARY_MAP, TRANSFORM_MAP


def source_identity(filepath):
    """Return the source-scene identity of an image, independent of transform.

    The dataset applies multiple post-processing transforms to the same source
    scene, encoding the transform as a filename prefix (e.g. ``transfer_`` /
    ``redigital_``). Stripping that prefix (and the extension) yields a key that
    is shared across an image's original / transmitted / redigitalized versions,
    so a group-aware split can keep all versions of a scene in the same split.
    """
    base = os.path.splitext(os.path.basename(filepath))[0].lower()
    base = re.sub(r"^(transfer_|redigital_|redigital|transfer)", "", base)
    return base


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


def _assign_group_splits(df, val_frac, test_frac, seed):
    """Group-aware, stratified train/val/test assignment at the SOURCE-SCENE level.

    Every image of a given `ident` (source scene, across all its transforms) is
    assigned to the same split, so no scene content leaks across splits. Splitting
    is done independently within each (binary_label, primary transform) stratum to
    keep the real/fake and transform proportions balanced across all three splits.
    """
    # One row per identity: its binary label and its dominant transform.
    idf = (df.groupby("ident")
             .agg(binary_label=("binary_label", "first"),
                  transform_label=("transform_label",
                                   lambda s: s.value_counts().index[0]))
             .reset_index())

    rng = np.random.RandomState(seed)
    ident_to_split = {}
    for _, stratum in idf.groupby(["binary_label", "transform_label"]):
        ids = stratum["ident"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i in ids[:n_test]:
            ident_to_split[i] = "test"
        for i in ids[n_test:n_test + n_val]:
            ident_to_split[i] = "val"
        for i in ids[n_test + n_val:]:
            ident_to_split[i] = "train"

    out = df.copy()
    out["split"] = out["ident"].map(ident_to_split)
    return out


def build_splits(original_train_dir, original_val_dir, test_subset_dir,
                 subset_per_class, seed, val_frac=0.15, test_frac=0.15, csv_path=None):
    """Construct a group-aware, leak-free train/val/test split.

    All labelled images (original + transmitted + redigitalized) are pooled, a
    source-scene `ident` is computed for each, and identities are split into
    train/val/test so that every transform version of a scene stays in one split
    (no cross-transform content leakage). The split is stratified by
    (binary_label, primary transform). A balanced per-class cap is then applied
    within each split. Returns (train_df, val_df, test_df).
    """
    print("\n--- Building group-aware train/val/test splits ---")
    orig_train = _scan_root(original_train_dir, transform_override="original")
    orig_val = _scan_root(original_val_dir, transform_override="original")
    originals = pd.concat([orig_train, orig_val], ignore_index=True)
    print(f"  original: {len(originals)} images")

    other = _scan_root(test_subset_dir)
    other = other[other["transform_label"].isin(["transmitted", "redigitalized"])].reset_index(drop=True)
    print(f"  transmitted+redigitalized (test_subset): {len(other)} images")

    pool = pd.concat([originals, other], ignore_index=True)
    pool["ident"] = pool["filepath"].map(source_identity)

    pool = _assign_group_splits(pool, val_frac=val_frac, test_frac=test_frac, seed=seed)

    # Balanced per-class cap within each split.
    def _balance(df):
        if not len(df):
            return df
        return pd.concat([
            grp.sample(min(len(grp), subset_per_class), random_state=seed)
            for _, grp in df.groupby(["binary_label", "transform_label"])
        ]).reset_index(drop=True)

    train_df = _balance(pool[pool["split"] == "train"])
    val_df = _balance(pool[pool["split"] == "val"])
    test_df = _balance(pool[pool["split"] == "test"])

    # Sanity 1: no file in two splits.
    paths = [set(d["filepath"]) for d in (train_df, val_df, test_df)]
    assert not (paths[0] & paths[1] | paths[0] & paths[2] | paths[1] & paths[2]), \
        "LEAK: a file appears in more than one split!"
    # Sanity 2: no source identity in two splits (group-aware guarantee).
    ids = [set(d["ident"]) for d in (train_df, val_df, test_df)]
    assert not (ids[0] & ids[1] | ids[0] & ids[2] | ids[1] & ids[2]), \
        "LEAK: a source identity appears in more than one split!"

    print(f"  -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)} images")

    if csv_path is not None:
        pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(csv_path, index=False)

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


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
# CLIP uses its own input normalization; using ImageNet stats for a CLIP backbone
# silently shifts the inputs off the distribution it was trained on.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def normalization_for(backbone):
    """Return (mean, std) appropriate for the given backbone."""
    if str(backbone).startswith("clip"):
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


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


def get_train_transform(img_size, backbone="resnet50"):
    """Forensics-friendly training augmentation: crop native pixels + degradations."""
    mean, std = normalization_for(backbone)
    return transforms.Compose([
        ResizeIfSmaller(img_size),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        # RandomJPEG(p=0.5, quality_range=(50, 95)),
        transforms.ToTensor(),
        # GaussianNoise(p=0.3, std=0.02),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transform(img_size, backbone="resnet50"):
    """Validation/test transform: aspect-preserving resize + center crop.

    Avoids the square `Resize((s, s))` distortion of the previous pipeline; only
    resizes the shorter side, then center-crops, minimizing resampling.
    """
    mean, std = normalization_for(backbone)
    return transforms.Compose([
        ResizeIfSmaller(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_tta_transform(img_size, backbone="resnet50", n_crops=5):
    """Multi-crop TTA transform for the final test evaluation.

    A single 224 center crop of a ~680px image discards >90% of its pixels.
    This returns a deterministic stack of native-resolution crops — 4 corners +
    center (FiveCrop), or those plus horizontal flips (TenCrop when n_crops>=10)
    — as a [n_crops, 3, img_size, img_size] tensor. Prediction collection
    averages the softmax over crops (see evaluation.collect_*_predictions).
    """
    mean, std = normalization_for(backbone)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)
    cropper = transforms.TenCrop(img_size) if n_crops >= 10 else transforms.FiveCrop(img_size)
    return transforms.Compose([
        ResizeIfSmaller(img_size),
        cropper,
        transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(c)) for c in crops])),
    ])
