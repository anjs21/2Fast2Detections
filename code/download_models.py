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

    # DRCT-ConvB backbone (the default, used by every stage via CONFIG["backbone"]).
    print("Downloading convnext_base weights (DRCT-ConvB backbone)...")
    models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

    # Remaining backbones supported by models.get_backbone() are pre-cached so any
    # CONFIG["backbone"] swap works offline (see config.py: backbone options).
    print("Downloading convnext_tiny weights...")
    models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

    # print("Downloading resnet18 weights...")
    # models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # print("Downloading resnet50 weights...")
    # models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # print("Downloading efficientnet_b0 weights...")
    # models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    print("Model backbones downloaded successfully!")
except Exception as e:
    print(f"Error downloading models: {e}")

print("\n=======================================================")
print("Pre-downloading DRCT Stable Diffusion weights...")
print("=======================================================")
# The DRCT reconstruction step (drct_reconstruct.py) runs Stable Diffusion img2img
# to mine "hard fakes". Pre-cache its weights so the first training run doesn't
# stall on a multi-GB download. Guarded: diffusers may be absent and this step is
# optional (training still runs with the contrastive loss on labelled data only).
try:
    from config import CONFIG

    if CONFIG.get("use_drct") and CONFIG.get("drct_reconstruct"):
        sd_model = CONFIG.get("drct_sd_model", "stabilityai/sd-turbo")
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline

            print(f"Downloading Stable Diffusion weights: {sd_model} ...")
            # Cache weights to the local HF hub without committing GPU memory.
            StableDiffusionImg2ImgPipeline.from_pretrained(
                sd_model,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            print("Stable Diffusion weights cached successfully!")
        except ImportError as e:
            print(f"diffusers/torch not available ({e}). "
                  f"Skipping SD pre-download; DRCT reconstruction will be skipped at runtime.")
    else:
        print("DRCT reconstruction disabled in CONFIG; skipping SD weights download.")
except Exception as e:
    print(f"Error downloading Stable Diffusion weights: {e}")

print("\n=======================================================")
print("All required models and datasets are pre-downloaded and cached!")
print("=======================================================")
