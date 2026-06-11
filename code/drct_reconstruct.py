"""
DRCT: Diffusion Reconstruction of real images into "hard fakes".

This implements the data-mining half of DRCT (Chen et al., ICML 2024). Real
training images are passed through a Stable Diffusion img2img reconstruction at
low noising strength, producing near-copies that nonetheless carry diffusion
generation artifacts. These reconstructions are added to the training set as
*fake* examples; the supervised-contrastive loss in training.py then learns the
subtle real-vs-generated boundary that gives DRCT-ConvB its robustness.

This step is OPTIONAL and guarded: it requires `diffusers` + a GPU + (first run)
a Stable Diffusion weights download. If any of those are missing it prints a
clear message and returns an empty DataFrame, and training proceeds on the
original labelled data with the contrastive loss still active.
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _reconstructions_present(recon_dir):
    if not os.path.isdir(recon_dir):
        return 0
    return sum(
        1 for root, _, files in os.walk(recon_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def build_drct_reconstructions(train_df, config, recon_dir):
    """Reconstruct real train images as hard-fakes. Returns rows to append to train_df."""
    if not (config.get("use_drct") and config.get("drct_reconstruct")):
        return pd.DataFrame()

    # Reuse a previous run's reconstructions if available.
    existing = _reconstructions_present(recon_dir)
    if existing > 0:
        print(f"[DRCT] Reusing {existing} existing reconstructions in {recon_dir}")
        return _scan_recon_dir(recon_dir)

    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
    except Exception as e:
        print(f"[DRCT] diffusers not available ({e}). "
              f"Skipping reconstruction; training will use labelled data + contrastive loss only.")
        return pd.DataFrame()

    device = config["device"]
    if device != "cuda":
        print("[DRCT] No GPU detected — SD reconstruction is impractical on CPU. "
              "Skipping reconstruction (contrastive loss still active).")
        return pd.DataFrame()

    print(f"[DRCT] Loading Stable Diffusion: {config['drct_sd_model']} ...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        config["drct_sd_model"], torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(recon_dir, exist_ok=True)
    real = train_df[train_df["binary_label"] == "real"]
    per_class = config.get("drct_recon_per_class", 500)
    strength = config.get("drct_recon_strength", 0.2)

    rows = []
    for transform_label, grp in real.groupby("transform_label"):
        grp = grp.sample(min(len(grp), per_class), random_state=config["seed"])
        out_subdir = os.path.join(recon_dir, transform_label)
        os.makedirs(out_subdir, exist_ok=True)
        for i, (_, row) in enumerate(tqdm(grp.iterrows(), total=len(grp),
                                          desc=f"[DRCT] reconstruct {transform_label}")):
            try:
                img = Image.open(row["filepath"]).convert("RGB").resize((512, 512))
                recon = pipe(prompt="", image=img, strength=strength,
                             guidance_scale=1.0, num_inference_steps=25).images[0]
                out_path = os.path.join(out_subdir, f"recon_{i:06d}.png")
                recon.save(out_path)
                rows.append({
                    "filepath": out_path,
                    "binary_label": "fake",            # reconstructed real = hard fake
                    "transform_label": transform_label,
                    "width": recon.width, "height": recon.height,
                    "is_corrupted": False, "split": "train", "is_drct_recon": True,
                })
            except Exception as e:
                print(f"[DRCT] skip {row['filepath']}: {e}")

    print(f"[DRCT] Built {len(rows)} reconstructed hard-fakes.")
    return pd.DataFrame(rows)


def _scan_recon_dir(recon_dir):
    rows = []
    for transform_label in os.listdir(recon_dir):
        sub = os.path.join(recon_dir, transform_label)
        if not os.path.isdir(sub):
            continue
        for f in os.listdir(sub):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append({
                    "filepath": os.path.join(sub, f),
                    "binary_label": "fake", "transform_label": transform_label,
                    "width": None, "height": None,
                    "is_corrupted": False, "split": "train", "is_drct_recon": True,
                })
    return pd.DataFrame(rows)
