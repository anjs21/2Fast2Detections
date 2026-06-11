"""
Computer Vision Project v2: Joint Detection of AI-Generated Images
and Post-Processing Alterations in Real-World Scenarios

Main entry point that orchestrates all stages:
  1. Data preparation
  2. Unimodal baselines (stage_unimodal)
  3. Multi-task joint training (stage_multimodal)
  4. Ablation study (stage_ablation)
  5. Evaluation & save results (stage_test_eval)

Improvements:
  - Leak-free splits: "original" honors the dataset's curated train/val folders;
    "transmitted"/"redigitalized" are split 80/20 from test_subset, disjoint at
    the image level (see data.build_splits). No image appears in both splits.
  - DRCT-ConvB: ConvNeXt-Base shared backbone + supervised-contrastive head, with
    optional Stable Diffusion reconstruction mining real images into hard-fakes
    (drct_reconstruct.py).
  - Forensics-aware augmentation: native-pixel crops + JPEG/noise degradations
    instead of resampling augmentation that erases generation artifacts.
  - Class-weighted cross-entropy, mixed-precision (AMP) training.
  - Task heads with Dropout + hidden layer.
  - LR scheduler (CosineAnnealingLR), early stopping, best-model checkpointing,
    gradient clipping.
  - Unimodal baselines, ablation over loss weights, per-transformation breakdown,
    cross-class trace analysis.
  - Metrics: F1, precision, recall, confusion matrices, ROC-AUC / AP for binary.
  - Training-curve visualization.
"""

import os
import time
import torch
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from torch.utils.data import DataLoader

from config import (
    CONFIG, DATA_DIR, METADATA_TRAIN_VAL_CSV, PROJECT_ROOT,
    ORIGINAL_TRAIN_DIR, ORIGINAL_VAL_DIR, TEST_SUBSET_DIR, DRCT_RECON_DIR,
    download_train_val_data, download_test_data,
)
from data import (
    build_splits, MultiTaskDataset, SingleTaskDataset,
    get_train_transform, get_val_transform,
)
from drct_reconstruct import build_drct_reconstructions
from stage_unimodal import run_unimodal_baselines
from stage_multimodal import run_multimodal_training, run_comparison_and_analysis
from stage_ablation import run_ablation_study
from stage_test_eval import save_results


if __name__ == "__main__":
    start_time = time.time()

    # ------------------------------------------------------------------
    # PHASE 1: DATA PREPARATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)

    # 1a. Download train/val data and test data
    download_train_val_data()
    download_test_data()

    # 1b-1e. Build LEAK-FREE train/val splits:
    #   - "original": honor the dataset's curated train/ and val/ folders as-is
    #   - "transmitted"/"redigitalized": 80/20 split of the test_subset images
    #     (disjoint at image level -> no train/val leak). A balanced per-class cap
    #     is applied within each split. See data.build_splits.
    train_df, val_df = build_splits(
        ORIGINAL_TRAIN_DIR, ORIGINAL_VAL_DIR, TEST_SUBSET_DIR,
        subset_per_class=CONFIG["subset_per_class"], seed=CONFIG["seed"],
        csv_path=METADATA_TRAIN_VAL_CSV,
    )

    # 1e-bis. DRCT: mine diffusion-reconstructed "hard fakes" from real TRAIN
    # images and append them to the training set (optional, GPU+diffusers; see
    # drct_reconstruct.py). val_df is left untouched so evaluation stays honest.
    if CONFIG.get("use_drct"):
        recon_df = build_drct_reconstructions(train_df, CONFIG, DRCT_RECON_DIR)
        if len(recon_df):
            keep = [c for c in train_df.columns if c in recon_df.columns]
            train_df = pd.concat([train_df, recon_df[keep]], ignore_index=True)
            print(f"Added {len(recon_df)} DRCT hard-fakes -> train size now {len(train_df)}")

    print(f"\nTrain subset size: {len(train_df)}")
    print(f"Validation subset size: {len(val_df)}")
    print("\nTraining balance matrix:")
    print(pd.crosstab(train_df["transform_label"], train_df["binary_label"]))

    # 1f. Hardware / subset documentation
    print(f"\n--- Hardware & Subset Documentation ---")
    print(f"Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Subset per class: {CONFIG['subset_per_class']}")
    print(f"Total training images: {len(train_df)}")
    print(f"Total validation images: {len(val_df)}")
    print(f"Backbone: {CONFIG['backbone']}")

    # 1g. Create DataLoaders
    train_tfm = get_train_transform(CONFIG["img_size"])
    val_tfm = get_val_transform(CONFIG["img_size"])

    # Multi-task dataloaders
    mt_train_dataset = MultiTaskDataset(train_df, transform=train_tfm)
    mt_val_dataset = MultiTaskDataset(val_df, transform=val_tfm)
    mt_train_loader = DataLoader(mt_train_dataset, batch_size=CONFIG["batch_size"],
                                 shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    mt_val_loader = DataLoader(mt_val_dataset, batch_size=CONFIG["batch_size"],
                                shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # Single-task dataloaders (for unimodal baselines)
    st_bin_train = DataLoader(
        SingleTaskDataset(train_df, "binary", train_tfm),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    st_bin_val = DataLoader(
        SingleTaskDataset(val_df, "binary", val_tfm),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    st_trans_train = DataLoader(
        SingleTaskDataset(train_df, "transform", train_tfm),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    st_trans_val = DataLoader(
        SingleTaskDataset(val_df, "transform", val_tfm),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True
    )

    # ------------------------------------------------------------------
    # PHASE 2: UNIMODAL BASELINES
    # ------------------------------------------------------------------
    unimodal_bin_acc, unimodal_trans_acc = run_unimodal_baselines(
        st_bin_train, st_bin_val, st_trans_train, st_trans_val
    )

    # ------------------------------------------------------------------
    # PHASE 3: MULTI-TASK JOINT TRAINING
    # ------------------------------------------------------------------
    model_mt, results, multitask_bin_acc, multitask_trans_acc = run_multimodal_training(
        mt_train_loader, mt_val_loader, val_df, val_tfm
    )

    # ------------------------------------------------------------------
    # PHASES 4-6, 8: COMPARISON, ANALYSIS, VISUAL INFERENCE
    # ------------------------------------------------------------------
    comparison_df, breakdown_df, trace_df = run_comparison_and_analysis(
        results, val_df, val_tfm, model_mt,
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc
    )

    # # ------------------------------------------------------------------
    # # PHASE 7: ABLATION STUDY
    # # ------------------------------------------------------------------
    # print("\n" + "=" * 60)
    # print("PHASE 7: ABLATION STUDY (Loss Weight Sweep)")
    # print("=" * 60)

    # ablation_df = run_ablation_study(mt_train_loader, mt_val_loader)

    # ------------------------------------------------------------------
    # PHASE 9: SAVE FINAL RESULTS
    # ------------------------------------------------------------------
    save_results(
        model_mt, start_time,
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc,
        ablation_df, breakdown_df, trace_df
    )
