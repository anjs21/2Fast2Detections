import os
import urllib.request
import tarfile
import shutil
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from globals import *

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"Downloading {output_path}") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_train_val_data():
    original_dir = os.path.join(DATA_DIR, "RRDataset_original_train_val")
    if os.path.exists(original_dir):
        print(f"Train/val data is already present in '{original_dir}'. Skipping download!")
        return

    tar_path = os.path.join(DATA_DIR, "..", "train_val.tar.gz")
    tar_path = os.path.abspath(tar_path)
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

def download_test_data():
    test_subset_dir = os.path.join(DATA_DIR, "test_subset")
    local_tar_path = os.path.join(DATA_DIR, "..", "test_subset.tar.gz")
    local_tar_path = os.path.abspath(local_tar_path)

    if os.path.exists(test_subset_dir):
        image_count = 0
        for root, dirs, files in os.walk(test_subset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_count += 1
        if image_count >= 3800:
            print(f"Test subset is already extracted ({image_count} images found). Skipping download!")
            return

    if not os.path.exists(local_tar_path):
        url = "https://zenodo.org/api/records/14963880/files/RRDataset_test.tar.gz/content"
        download_url(url, local_tar_path)

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

            if is_redigital and is_ai and count_redigital_ai < LIMIT:
                tar.extract(member, path=test_subset_dir)
                count_redigital_ai += 1

            if all(c >= LIMIT for c in counts.values()) and count_redigital_ai >= LIMIT:
                break

    print(f"Extraction complete! Final counts:")
    print(f"  - redigital_real: {counts['redigital_real']}")
    print(f"  - transmitted_real: {counts['transmitted_real']}")
    print(f"  - transmitted_ai: {counts['transmitted_ai']}")
    print(f"  - redigital_ai: {count_redigital_ai}")

class EarlyStopping:
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
    def __init__(self):
        self.history = defaultdict(list)

    def log(self, epoch, **kwargs):
        self.history["epoch"].append(epoch)
        for k, v in kwargs.items():
            self.history[k].append(v)

    def to_dataframe(self):
        return pd.DataFrame(self.history)

def compute_class_weights(df, col, label_map, device):
    counts = torch.zeros(len(label_map), dtype=torch.float)
    vc = df[col].map(label_map).value_counts()
    for idx, n in vc.items():
        counts[int(idx)] = n
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(label_map) * counts)
    return weights.to(device)
