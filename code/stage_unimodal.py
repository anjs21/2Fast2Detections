"""
Stage: Unimodal Baselines

Trains two independent single-task models:
  1. Binary classifier (Real vs Fake)
  2. Transformation classifier (Original / Transmitted / Redigitalized)
"""

from config import CONFIG
from models import SingleTaskModel, DualStreamSingleTaskModel
from training import train_singletask_model
from evaluation import (
    collect_singletask_predictions,
    full_classification_report,
    plot_training_curves,
)


def _build_singletask_model(num_classes, head_cfg):
    """Same architecture selection as the multi-task stages: when dual_stream is
    enabled the baselines also use the RGB + Bayar noise-residual model, and each
    baseline uses its matching task head (head_cfg), so the joint-vs-unimodal
    comparison holds the architecture fixed and isolates multi-task training."""
    if CONFIG.get("dual_stream"):
        return DualStreamSingleTaskModel(
            rgb_backbone_name=CONFIG["backbone"],
            noise_backbone_name=CONFIG.get("noise_backbone", "resnet18"),
            num_classes=num_classes,
            trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
            dropout=CONFIG["dropout"],
            head_cfg=head_cfg,
        ).to(CONFIG["device"])
    return SingleTaskModel(
        backbone_name=CONFIG["backbone"], num_classes=num_classes,
        trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
        dropout=CONFIG["dropout"],
        head_cfg=head_cfg,
    ).to(CONFIG["device"])


def run_unimodal_baselines(st_bin_train, st_bin_val, st_bin_test,
                           st_trans_train, st_trans_val, st_trans_test):
    """
    Train and evaluate both unimodal baselines.

    Each model is fit on train, selected/early-stopped on val, and finally
    evaluated on the held-out test set (the reported numbers).

    Returns:
        unimodal_bin_acc: Test accuracy for binary baseline
        unimodal_trans_acc: Test accuracy for transform baseline
    """
    print("\n" + "=" * 60)
    print("PHASE 2: UNIMODAL BASELINES")
    print("=" * 60)

    if CONFIG.get("dual_stream"):
        print(f"[model] Unimodal baselines use dual-stream: RGB={CONFIG['backbone']} "
              f"+ Bayar noise stream={CONFIG.get('noise_backbone', 'resnet18')}")

    # ---- 2a. Binary-only baseline (same head as the multi-task binary head) ----
    model_bin_only = _build_singletask_model(num_classes=2, head_cfg=CONFIG.get("binary_head"))
    model_bin_only, logger_bin = train_singletask_model(
        model_bin_only, st_bin_train, st_bin_val, CONFIG, task_name="binary"
    )
    plot_training_curves(logger_bin, "Unimodal Baseline: Binary (Real/Fake)",
                         "curves_unimodal_binary.png")

    # Final evaluation on the held-out TEST set
    preds_bin_uni, labels_bin_uni = collect_singletask_predictions(
        model_bin_only, st_bin_test, CONFIG["device"]
    )
    unimodal_bin_acc = (preds_bin_uni == labels_bin_uni).mean()
    full_classification_report(preds_bin_uni, labels_bin_uni,
                               ["Real", "Fake"], "Unimodal Binary Baseline (TEST)")

    # ---- 2b. Transform-only baseline (same head as the multi-task transform head) ----
    model_trans_only = _build_singletask_model(num_classes=3, head_cfg=CONFIG.get("transform_head"))
    model_trans_only, logger_trans = train_singletask_model(
        model_trans_only, st_trans_train, st_trans_val, CONFIG, task_name="transform"
    )
    plot_training_curves(logger_trans, "Unimodal Baseline: Transformation Type",
                         "curves_unimodal_transform.png")

    # Final evaluation on the held-out TEST set
    preds_trans_uni, labels_trans_uni = collect_singletask_predictions(
        model_trans_only, st_trans_test, CONFIG["device"]
    )
    unimodal_trans_acc = (preds_trans_uni == labels_trans_uni).mean()
    full_classification_report(preds_trans_uni, labels_trans_uni,
                               ["Original", "Transmitted", "Redigitalized"],
                               "Unimodal Transform Baseline (TEST)")

    return unimodal_bin_acc, unimodal_trans_acc
