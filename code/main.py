"""
Computer Vision Project v2: Joint Detection of AI-Generated Images
and Post-Processing Alterations in Real-World Scenarios

Main entry point that orchestrates all stages:
  1. Data preparation
  2. Unimodal baselines (stage_unimodal)
  3. Multi-task joint training (stage_multimodal)
  4. Ablation study (stage_ablation)
  5. Evaluation & save results (stage_test_eval)

Improvements over v1:
  - Fixed test data leak (separate metadata for train/val vs. test)
  - Upgraded backbone: ResNet50 (configurable)
  - Better task heads: Dropout + hidden layer
  - Richer data augmentation pipeline
  - Learning rate scheduler (CosineAnnealingLR)
  - Early stopping + best-model checkpointing
  - Gradient clipping
  - Unimodal baseline training for comparison
  - Ablation study over loss weight configurations
  - Per-transformation accuracy breakdown
  - Cross-class transformation trace analysis
  - Full metrics: F1, precision, recall, confusion matrices
  - Training curve visualization
"""

import os
import time
import torch
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import (
    CONFIG, DATA_DIR, METADATA_TRAIN_VAL_CSV, PROJECT_ROOT,
    download_train_val_data, download_test_data,
)
from data import (
    scan_directory, MultiTaskDataset, SingleTaskDataset,
    get_train_transform, get_val_transform,
)
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

    # 1b. Scan DATA_DIR for all categories
    if os.path.exists(METADATA_TRAIN_VAL_CSV):
        print(f"Loading existing metadata from {METADATA_TRAIN_VAL_CSV}")
        df_train_val = pd.read_csv(METADATA_TRAIN_VAL_CSV)
        
        # Ensure relative paths in cached metadata are resolved to absolute using PROJECT_ROOT
        def make_absolute(path):
            if not os.path.isabs(path):
                return os.path.abspath(os.path.join(PROJECT_ROOT, path))
            return path
        df_train_val["filepath"] = df_train_val["filepath"].apply(make_absolute)

        # Check if the loaded metadata contains all 3 transformation categories
        available_classes = set(df_train_val["transform_label"].dropna().unique())
        if len(available_classes.intersection({"original", "transmitted", "redigitalized"})) < 3:
            print("Cached metadata is incomplete (missing classes). Re-scanning...")
            df_train_val = scan_directory(DATA_DIR, METADATA_TRAIN_VAL_CSV)
    else:
        df_train_val = scan_directory(DATA_DIR, METADATA_TRAIN_VAL_CSV)

    # 1c. Filter out corrupted/unknown
    df_clean = df_train_val[
        (df_train_val["is_corrupted"] == False) &
        (df_train_val["binary_label"] != "unknown") &
        (df_train_val["transform_label"] != "unknown")
    ]

    # 1d. Balanced subset
    balanced_df = pd.concat([
        grp.sample(min(len(grp), CONFIG["subset_per_class"]), random_state=CONFIG["seed"])
        for _, grp in df_clean.groupby(["binary_label", "transform_label"])
    ]).reset_index(drop=True)

    # 1e. Stratified 80/20 split
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=0.2,
        stratify=balanced_df[["binary_label", "transform_label"]],
        random_state=CONFIG["seed"],
    )

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
                                 shuffle=True, num_workers=2, pin_memory=True)
    mt_val_loader = DataLoader(mt_val_dataset, batch_size=CONFIG["batch_size"],
                                shuffle=False, num_workers=2, pin_memory=True)

    # Single-task dataloaders (for unimodal baselines)
    st_bin_train = DataLoader(
        SingleTaskDataset(train_df, "binary", train_tfm),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    st_bin_val = DataLoader(
        SingleTaskDataset(val_df, "binary", val_tfm),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True
    )
    st_trans_train = DataLoader(
        SingleTaskDataset(train_df, "transform", train_tfm),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    st_trans_val = DataLoader(
        SingleTaskDataset(val_df, "transform", val_tfm),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True
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

    # ------------------------------------------------------------------
    # PHASE 7: ABLATION STUDY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 7: ABLATION STUDY (Loss Weight Sweep)")
    print("=" * 60)

    ablation_df = run_ablation_study(mt_train_loader, mt_val_loader)

    # ------------------------------------------------------------------
    # PHASE 9: SAVE FINAL RESULTS
    # ------------------------------------------------------------------
    save_results(
        model_mt, start_time,
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc,
        ablation_df, breakdown_df, trace_df
    )
