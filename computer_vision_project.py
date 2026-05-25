"""
Converted from computer-vision-project.ipynb
"""

# %%
import torch
import os
import urllib.request
import tarfile
from pathlib import Path

print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Run nvidia-smi if available
if torch.cuda.is_available():
    os.system("nvidia-smi")

# Configure directories based on whether running in Kaggle or locally
KAGGLE_MODE = os.path.exists("/kaggle")
DATA_DIR = "/kaggle/working/data" if KAGGLE_MODE else "./data"
METADATA_CSV = "/kaggle/working/metadata.csv" if KAGGLE_MODE else "./metadata.csv"

# %%
# 1. Download using the exact API link
original_dir = os.path.join(DATA_DIR, "original")
if os.path.exists(original_dir):
    print(f"Train/val data is already present in '{original_dir}'. Skipping download!")
else:
    if not os.path.exists("train_val.tar.gz"):
        print("Downloading train_val.tar.gz...")
        urllib.request.urlretrieve(
            "https://zenodo.org/api/records/14963880/files/RRDataset_original_train_val.tar.gz/content",
            "train_val.tar.gz"
        )

    # 2. Re-create directory and extract
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(original_dir):
        print("Extracting train_val.tar.gz...")
        with tarfile.open("train_val.tar.gz", "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

# 3. View the top-level folders
if os.path.exists(DATA_DIR):
    print("Top level folders in data directory:", os.listdir(DATA_DIR))

# %%
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = Path(DATA_DIR)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
rows = []

all_paths = list(DATASET_ROOT.rglob("*"))

print("Scanning images and extracting metadata...")
for img_path in tqdm(all_paths):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue

    # Get lowercase path parts
    parts = [p.lower() for p in img_path.relative_to(DATASET_ROOT).parts]

    # FIX: Use substring matching (checking if keyword is INSIDE any part)
    if any("real" in p for p in parts):
        binary_label = "real"
    elif any("ai" in p or "fake" in p for p in parts) :
        binary_label = "fake"
    else:
        binary_label = "unknown"

    # FIX: Use substring matching for transformations
    if any("original" in p for p in parts):
        transform_label = "original"
    elif any(k in p for p in parts for k in ["transfer", "transmit", "internet", "social"]):
        transform_label = "transmitted"
    elif any(k in p for p in parts for k in ["redigital", "screen"]):
        transform_label = "redigitalized"
    else:
        transform_label = "unknown"

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
        "is_corrupted": is_corrupted
    })

df = pd.DataFrame(rows)
df.to_csv(METADATA_CSV, index=False)

print(f"\nScan Complete! Total images indexed: {len(df)}")
print("\nJoint Multi-Task Balance Matrix:")
print(pd.crosstab(df["transform_label"], df["binary_label"]))

# %%
import torch

CONFIG = {
    "seed": 42,
    "img_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-4,
    "subset_per_class": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load inventory and drop corrupted OR unknown rows
df = pd.read_csv(METADATA_CSV)

# CRITICAL FIX: Filter out "unknown" labels
df = df[
    (df["is_corrupted"] == False) & 
    (df["binary_label"] != "unknown") & 
    (df["transform_label"] != "unknown")
]

# 2. Sample a balanced subset based on CONFIG
balanced_df = (
    df.groupby(["binary_label", "transform_label"])
    .apply(lambda x: x.sample(min(len(x), CONFIG["subset_per_class"]), random_state=CONFIG["seed"]))
    .reset_index(drop=True)
)

# 3. Create a clean 80/20 Train/Validation Split
train_df, val_df = train_test_split(
    balanced_df, 
    test_size=0.2, 
    stratify=balanced_df[["binary_label", "transform_label"]], 
    random_state=CONFIG["seed"]
)

print(f"Cleaned Train subset size: {len(train_df)}")
print(f"Cleaned Validation subset size: {len(val_df)}")

# %%
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MultiTaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        # Map text labels to integers
        self.binary_map = {"real": 0, "fake": 1}
        self.transform_map = {"original": 0, "transmitted": 1, "redigitalized": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        # Get labels and convert to torch tensors
        label_bin = torch.tensor(self.binary_map[row["binary_label"]], dtype=torch.long)
        label_trans = torch.tensor(self.transform_map[row["transform_label"]], dtype=torch.long)
        
        return img, label_bin, label_trans

# %%
from torch.utils.data import DataLoader

# Standard ImageNet preprocessing
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(), # Basic augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate Datasets
train_dataset = MultiTaskDataset(train_df, transform=train_transform)
val_dataset = MultiTaskDataset(val_df, transform=val_transform)

# Instantiate DataLoaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

# %%
import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        # Load a lightweight, powerful pre-trained backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Find the number of input features to the original final layer
        num_features = self.backbone.fc.in_features
        
        # Remove the original classification layer
        self.backbone.fc = nn.Identity()
        
        # Create Task Head 1: Binary (Real vs Fake -> 2 classes)
        self.binary_head = nn.Linear(num_features, 2)
        
        # Create Task Head 2: Multi-class (Original vs Transmitted vs Redigitalized -> 3 classes)
        self.transform_head = nn.Linear(num_features, 3)

    def forward(self, x):
        # Extract shared features
        features = self.backbone(x)
        
        # Pass features through both independent heads
        out_binary = self.binary_head(features)
        out_transform = self.transform_head(features)
        
        return out_binary, out_transform

# %%
model = MultiTaskResNet()
model = model.to(CONFIG["device"])
print("Model initialized on:", CONFIG["device"])

# %%
import torch.optim as optim

# 1. Loss Functions and Optimizer
criterion_binary = nn.CrossEntropyLoss()
criterion_transform = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])

# Loss weights (Ablation study targets)
w1, w2 = 0.5, 0.5

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_bin, correct_trans = 0, 0
    total = 0
    
    for images, labels_bin, labels_trans in dataloader:
        images = images.to(device)
        labels_bin = labels_bin.to(device)
        labels_trans = labels_trans.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        out_bin, out_trans = model(images)
        
        # Compute individual losses
        loss_bin = criterion_binary(out_bin, labels_bin)
        loss_trans = criterion_transform(out_trans, labels_trans)
        
        # Joint Loss Function
        total_loss = (w1 * loss_bin) + (w2 * loss_trans)
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += total_loss.item() * images.size(0)
        _, pred_bin = torch.max(out_bin, 1)
        _, pred_trans = torch.max(out_trans, 1)
        
        correct_bin += (pred_bin == labels_bin).sum().item()
        correct_trans += (pred_trans == labels_trans).sum().item()
        total += images.size(0)
        
    epoch_loss = running_loss / total
    acc_bin = correct_bin / total
    acc_trans = correct_trans / total
    
    return epoch_loss, acc_bin, acc_trans

# %%
def evaluate_model(model, dataloader, device):
    model.eval()
    correct_bin, correct_trans, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels_bin, labels_trans in dataloader:
            images = images.to(device)
            labels_bin = labels_bin.to(device)
            labels_trans = labels_trans.to(device)
            
            out_bin, out_trans = model(images)
            
            _, pred_bin = torch.max(out_bin, 1)
            _, pred_trans = torch.max(out_trans, 1)
            
            correct_bin += (pred_bin == labels_bin).sum().item()
            correct_trans += (pred_trans == labels_trans).sum().item()
            total += images.size(0)
            
    return correct_bin / total, correct_trans / total

# --- Main Training Loop ---
print("Starting Training...")
for epoch in range(CONFIG["epochs"]):
    # Train
    train_loss, train_acc_bin, train_acc_trans = train_epoch(
        model, train_loader, optimizer, CONFIG["device"]
    )
    
    # Validate
    val_acc_bin, val_acc_trans = evaluate_model(model, val_loader, CONFIG["device"])
    
    # Print progress
    print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}] Loss: {train_loss:.4f}")
    print(f"  [Train] Real/Fake Acc: {train_acc_bin*100:.2f}% | Transform Acc: {train_acc_trans*100:.2f}%")
    print(f"  [Val]   Real/Fake Acc: {val_acc_bin*100:.2f}% | Transform Acc: {val_acc_trans*100:.2f}%")

print("\nTraining complete!")

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def test_and_report(model, dataloader, device):
    model.eval()
    all_preds_bin = []
    all_labels_bin = []
    
    with torch.no_grad():
        for images, labels_bin, _ in dataloader:
            images = images.to(device)
            out_bin, _ = model(images)
            _, preds = torch.max(out_bin, 1)
            
            all_preds_bin.extend(preds.cpu().numpy())
            all_labels_bin.extend(labels_bin.numpy())
            
    # Print metrics
    print("--- Real/Fake Classification Report ---")
    print(classification_report(all_labels_bin, all_preds_bin, target_names=["Real", "Fake"]))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels_bin, all_preds_bin)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Real/Fake Baseline Confusion Matrix')
    plt.show()

# Run it on your validation loader
test_and_report(model, val_loader, CONFIG["device"])

# %%
import matplotlib.pyplot as plt
from PIL import Image
import torch

def run_visual_inference(model, df, num_samples=3):
    model.eval()
    
    # Reverse mappings to convert integers back to text strings
    inv_binary_map = {0: "real", 1: "fake"}
    inv_transform_map = {0: "original", 1: "transmitted", 2: "redigitalized"}
    
    # Pick random sample rows from your validation dataframe
    samples = df.sample(num_samples, random_state=42)
    
    plt.figure(figsize=(15, 5))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = row["filepath"]
        true_bin = row["binary_label"]
        true_trans = row["transform_label"]
        
        # 1. Load the raw image for plotting
        raw_img = Image.open(img_path).convert("RGB")
        
        # 2. Preprocess the image for the model (add batch dimension with unsqueeze)
        input_tensor = val_transform(raw_img).unsqueeze(0).to(CONFIG["device"])
        
        # 3. Model Prediction
        with torch.no_grad():
            out_bin, out_trans = model(input_tensor)
            _, pred_bin = torch.max(out_bin, 1)
            _, pred_trans = torch.max(out_trans, 1)
            
        p_bin = inv_binary_map[pred_bin.item()]
        p_trans = inv_transform_map[pred_trans.item()]
        
        # 4. Visualization Setup
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(raw_img)
        
        # Green title if real/fake prediction matches truth, red if it fails
        title_color = "green" if p_bin == true_bin else "red"
        
        plt.title(
            f"True: {true_bin} ({true_trans})\n"
            f"Pred: {p_bin} ({p_trans})", 
            color=title_color, 
            fontsize=12
        )
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

# Execute inference on 3 images from your validation set
run_visual_inference(model, val_df, num_samples=3)

# %%
import tarfile
import urllib.request
import os
import shutil

# Clear the old unbalanced test folder
shutil.rmtree(os.path.join(DATA_DIR, "test_subset"), ignore_errors=True)

url = "https://zenodo.org/api/records/14963880/files/RRDataset_test.tar.gz/content"
output_dir = os.path.join(DATA_DIR, "test_subset")

# Define targets and counters
counts = {
    "redigital_real": 0,
    "transmitted_real": 0,
    "transmitted_ai": 0
}
LIMIT = 1000 

print("Streaming and targeted extracting... This will skip unwanted files.")

with urllib.request.urlopen(url) as response:
    with tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg'))):
                continue
                
            parts = member.name.lower().split('/')
            
            # Identify the specific missing categories
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
                
            # If it's a category we need and we haven't hit the limit, extract it
            if key and counts[key] < LIMIT:
                tar.extract(member, path=output_dir)
                counts[key] += 1
                
                if sum(counts.values()) % 500 == 0:
                    print(f"Progress -> Redigital_Real: {counts['redigital_real']}, Transmitted_Real: {counts['transmitted_real']}, Transmitted_AI: {counts['transmitted_ai']}")
            
            # Stop early if all targets are met
            if all(c >= LIMIT for c in counts.values()):
                print("All missing categories successfully balanced!")
                break

print("\nExtraction finished. Final counts:", counts)

# %%
from pathlib import Path
import os

# Check the subfolders inside the new test_subset directory
test_subset_path = Path(DATA_DIR) / "test_subset"

print("Folders extracted inside test_subset:")
for root, dirs, files in os.walk(test_subset_path):
    # Only print directories that actually contain files to keep it clean
    if files:
        relative_path = Path(root).relative_to(test_subset_path)
        print(f"  - {relative_path} ({len(files)} images)")

# %%
import tarfile
import urllib.request
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 1. Grab the final missing category: redigital_ai
url = "https://zenodo.org/api/records/14963880/files/RRDataset_test.tar.gz/content"
output_dir = os.path.join(DATA_DIR, "test_subset")
count = 0

print("Pulling the final 1000 redigital_ai images...")
with urllib.request.urlopen(url) as response:
    with tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                parts = member.name.lower().split('/')
                if "redigital" in parts and ("ai" in parts or "fake" in parts):
                    if count < 1000:
                        tar.extract(member, path=output_dir)
                        count += 1
                    else:
                        break

# 2. Re-run the Metadata Scanner over EVERYTHING
DATASET_ROOT = Path(DATA_DIR)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
rows = []

print("\nScanning entire dataset directory...")
for img_path in tqdm(list(DATASET_ROOT.rglob("*"))):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue

    parts = [p.lower() for p in img_path.relative_to(DATASET_ROOT).parts]

    # Binary Label
    if any("real" in p for p in parts):
        binary_label = "real"
    elif any("ai" in p or "fake" in p for p in parts):
        binary_label = "fake"
    else:
        binary_label = "unknown"

    # Transformation Label
    if any("original" in p for p in parts):
        transform_label = "original"
    elif any(k in p for p in parts for k in ["transfer", "transmit", "internet"]):
        transform_label = "transmitted"
    elif any(k in p for p in parts for k in ["redigital", "screen"]):
        transform_label = "redigitalized"
    else:
        transform_label = "unknown"

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
        "is_corrupted": is_corrupted
    })

df = pd.DataFrame(rows)
df.to_csv(METADATA_CSV, index=False)

print("\n=== NEW MULTI-TASK BALANCE MATRIX ===")
print(pd.crosstab(df["transform_label"], df["binary_label"]))

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# 1. Filter out corrupted/unknown images
df = pd.read_csv(METADATA_CSV)
df = df[(df["is_corrupted"] == False) & (df["binary_label"] != "unknown") & (df["transform_label"] != "unknown")]

# 2. Create the perfectly balanced subset (1000 per combination)
balanced_df = (
    df.groupby(["binary_label", "transform_label"])
    .apply(lambda x: x.sample(min(len(x), CONFIG["subset_per_class"]), random_state=CONFIG["seed"]))
    .reset_index(drop=True)
)

# 3. Stratified 80/20 split
train_df, val_df = train_test_split(
    balanced_df, 
    test_size=0.2, 
    stratify=balanced_df[["binary_label", "transform_label"]], 
    random_state=CONFIG["seed"]
)

# 4. Refresh DataLoaders
train_dataset = MultiTaskDataset(train_df, transform=train_transform)
val_dataset = MultiTaskDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

print(f"Train subset size: {len(train_df)}")
print(f"Validation subset size: {len(val_df)}")
print("\nNew Joint Multi-Task Training Balance:")
print(pd.crosstab(train_df["transform_label"], train_df["binary_label"]))

# %%
import torch.optim as optim

# Initialize fresh model and optimizer
model = MultiTaskResNet().to(CONFIG["device"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])

print(f"Starting true multi-task training on {CONFIG['device']}...")

for epoch in range(CONFIG["epochs"]):
    # Train
    train_loss, train_acc_bin, train_acc_trans = train_epoch(
        model, train_loader, optimizer, CONFIG["device"]
    )
    
    # Validate
    val_acc_bin, val_acc_trans = evaluate_model(model, val_loader, CONFIG["device"])
    
    # Report progress
    print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}] Joint Loss: {train_loss:.4f}")
    print(f"  [Train] Real/Fake: {train_acc_bin*100:.2f}% | Transform: {train_acc_trans*100:.2f}%")
    print(f"  [Val]   Real/Fake: {val_acc_bin*100:.2f}% | Transform: {val_acc_trans*100:.2f}%")

print("\nMulti-task baseline training complete!")

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def evaluate_multitask_performance(model, dataloader, device):
    model.eval()
    all_preds_bin, all_labels_bin = [], []
    all_preds_trans, all_labels_trans = [], []
    
    with torch.no_grad():
        for images, labels_bin, labels_trans in dataloader:
            images = images.to(device)
            out_bin, out_trans = model(images)
            
            all_preds_bin.extend(torch.max(out_bin, 1)[1].cpu().numpy())
            all_labels_bin.extend(labels_bin.numpy())
            
            all_preds_trans.extend(torch.max(out_trans, 1)[1].cpu().numpy())
            all_labels_trans.extend(labels_trans.numpy())
            
    # Plotting Matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Task 1: Real/Fake Matrix
    cm_bin = confusion_matrix(all_labels_bin, all_preds_bin)
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    axes[0].set_title("Real/Fake Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")
    
    # Task 2: Transformation Matrix
    cm_trans = confusion_matrix(all_labels_trans, all_preds_trans)
    sns.heatmap(cm_trans, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=["Original", "Transmitted", "Redigitalized"],
                yticklabels=["Original", "Transmitted", "Redigitalized"])
    axes[1].set_title("Transformation Confusion Matrix")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")
    
    plt.tight_layout()
    plt.show()
    
    # Print Metrics Reports
    print("="*25 + " REAL/FAKE REPORT " + "="*25)
    print(classification_report(all_labels_bin, all_preds_bin, target_names=["Real", "Fake"]))
    
    print("\n" + "="*23 + " TRANSFORMATION REPORT " + "="*23)
    print(classification_report(all_labels_trans, all_preds_trans, target_names=["Original", "Transmitted", "Redigitalized"]))

# Run metrics evaluation
evaluate_multitask_performance(model, val_loader, CONFIG["device"])

# %%
import numpy as np

def run_multitask_inference(model, df, num_samples=3):
    model.eval()
    inv_binary_map = {0: "real", 1: "fake"}
    inv_transform_map = {0: "original", 1: "transmitted", 2: "redigitalized"}
    
    samples = df.sample(num_samples, random_state=np.random.randint(1, 1000))
    plt.figure(figsize=(16, 5))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        raw_img = Image.open(row["filepath"]).convert("RGB")
        input_tensor = val_transform(raw_img).unsqueeze(0).to(CONFIG["device"])
        
        with torch.no_grad():
            out_bin, out_trans = model(input_tensor)
            p_bin = inv_binary_map[torch.max(out_bin, 1)[1].item()]
            p_trans = inv_transform_map[torch.max(out_trans, 1)[1].item()]
            
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(raw_img)
        
        # True vs Predicted comparison
        is_correct = (p_bin == row["binary_label"]) and (p_trans == row["transform_label"])
        title_color = "green" if is_correct else "red"
        
        plt.title(
            f"True: {row['binary_label']} | {row['transform_label']}\n"
            f"Pred: {p_bin} | {p_trans}", 
            color=title_color, 
            fontsize=11
        )
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()

# Run visual inference on 3 random validation images
run_multitask_inference(model, val_df, num_samples=3)

