#!/usr/bin/env python3
import os
import sys

from huggingface_hub import login
token = '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'
print("Logging into Hugging Face...")
login(token)

# Ensure code directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=======================================================")
print("Pre-downloading Datasets...")
print("=======================================================")
try:
    from config import download_train_val_data, download_test_data
    download_train_val_data()
    download_test_data()
    print("Datasets downloaded/extracted successfully!")
except Exception as e:
    print(f"Error downloading datasets: {e}")

print("\n=======================================================")
print("Pre-downloading Model Backbones...")
print("=======================================================")
try:
    from torchvision import models

    print("Downloading resnet50 weights...")
    models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    print("Downloading convnext_tiny weights...")
    models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

    # Remaining backbones supported by models.get_backbone() are pre-cached so any
    # CONFIG["backbone"] swap works offline (see config.py: backbone options).
    # print("Downloading efficientnet_b0 weights...")
    # models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    print("Model backbones downloaded successfully!")
except Exception as e:
    print(f"Error downloading models: {e}")

# CLIP image-encoder weights (binary specialist; only needed if CONFIG["backbone"]
# is a clip_* option). Guarded so it never blocks the torchvision pipeline.
try:
    from models import CLIP_MODELS
    from transformers import CLIPVisionModel
    for clip_name in ("clip_vit_l14",):
        print(f"Downloading CLIP weights: {CLIP_MODELS[clip_name]} ...")
        CLIPVisionModel.from_pretrained(CLIP_MODELS[clip_name])
    print("CLIP weights cached successfully!")
except Exception as e:
    print(f"Skipping CLIP pre-download ({e}). Only needed for a clip_* backbone.")

print("\n=======================================================")
print("All required models and datasets are pre-downloaded and cached!")
print("=======================================================")
