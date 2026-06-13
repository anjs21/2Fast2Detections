import io
import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from globals import *

def source_identity(filepath):
    """Computes the identity of the source scene from the filename."""
    base = os.path.splitext(os.path.basename(filepath))[0].lower()
    base = re.sub(r"^(transfer_|redigital_|redigital|transfer)", "", base)
    return base

def assign_labels(parts):
    """Assigns labels based on directory structure."""
    if any("real" in p for p in parts):
        binary_label = "real"
    elif any("ai" in p or "fake" in p for p in parts):
        binary_label = "fake"
    else:
        binary_label = "unknown"

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

    paths = [set(d["filepath"]) for d in (train_df, val_df, test_df)]
    assert not (paths[0] & paths[1] | paths[0] & paths[2] | paths[1] & paths[2]), \
        "LEAK: a file appears in more than one split!"
    ids = [set(d["ident"]) for d in (train_df, val_df, test_df)]
    assert not (ids[0] & ids[1] | ids[0] & ids[2] | ids[1] & ids[2]), \
        "LEAK: a source identity appears in more than one split!"

    print(f"  -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)} images")
    if csv_path is not None:
        pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(csv_path, index=False)

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))

class MultiTaskDataset(Dataset):
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
    def __init__(self, df, task, transform=None):
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def normalization_for(backbone):
    if str(backbone).startswith("clip"):
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD

class RandomJPEG:
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
    def __init__(self, p=0.3, std=0.02):
        self.p = p
        self.std = std

    def __call__(self, t):
        if torch.rand(1).item() >= self.p:
            return t
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)

class ResizeIfSmaller:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if min(w, h) < self.size:
            return transforms.functional.resize(img, self.size)
        return img

def get_train_transform(img_size, backbone="resnet50"):
    mean, std = normalization_for(backbone)
    return transforms.Compose([
        ResizeIfSmaller(img_size),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_val_transform(img_size, backbone="resnet50"):
    mean, std = normalization_for(backbone)
    return transforms.Compose([
        ResizeIfSmaller(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_tta_transform(img_size, backbone="resnet50", n_crops=5):
    mean, std = normalization_for(backbone)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)
    cropper = transforms.TenCrop(img_size) if n_crops >= 10 else transforms.FiveCrop(img_size)
    return transforms.Compose([
        ResizeIfSmaller(img_size),
        cropper,
        transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(c)) for c in crops])),
    ])
