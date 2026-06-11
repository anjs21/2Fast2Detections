"""
Stage: Unimodal Baselines

Trains two independent single-task models:
  1. Binary classifier (Real vs Fake)
  2. Transformation classifier (Original / Transmitted / Redigitalized)
"""

from config import CONFIG
from models import SingleTaskModel
from training import train_singletask_model
from evaluation import (
    collect_singletask_predictions,
    full_classification_report,
    plot_training_curves,
)


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

    # ---- 2a. Binary-only baseline ----
    model_bin_only = SingleTaskModel(
        backbone_name=CONFIG["backbone"], num_classes=2,
        trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
    ).to(CONFIG["device"])
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

    # ---- 2b. Transform-only baseline ----
    model_trans_only = SingleTaskModel(
        backbone_name=CONFIG["backbone"], num_classes=3,
        trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
    ).to(CONFIG["device"])
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
