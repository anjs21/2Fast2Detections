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

    # Default backbone (used by every stage via CONFIG["backbone"]).
    print("Downloading resnet50 weights...")
    models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Remaining backbones supported by models.get_backbone() are pre-cached so any
    # CONFIG["backbone"] swap works offline (see config.py: backbone options).
    # print("Downloading resnet18 weights...")
    # models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # print("Downloading efficientnet_b0 weights...")
    # models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    print("Model backbones downloaded successfully!")
except Exception as e:
    print(f"Error downloading models: {e}")

print("\n=======================================================")
print("All required models and datasets are pre-downloaded and cached!")
print("=======================================================")
