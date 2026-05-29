"""
Stage: Ablation Study

Sweeps different loss weight configurations (w1, w2) for the
multi-task model to study how the two tasks interact.
"""

import os
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from config import CONFIG, RESULTS_DIR
from models import MultiTaskModel
from training import (train_multitask_model, evaluate_multitask,
                      StaticWeighter, UncertaintyLoss, GradNormBalancer)


def run_ablation_study(train_loader, val_loader):
    """
    Sweep different loss weight configurations and compare task accuracies.
    Tests whether the two tasks compete or complement each other.

    Returns:
        ablation_df: DataFrame with results for each weight configuration
    """
    weight_configs = [
        ("static",      1.0,  0.0,  "binary_only"),
        ("static",      0.75, 0.25, "binary_dominant"),
        ("static",      0.5,  0.5,  "equal"),
        ("static",      0.25, 0.75, "transform_dominant"),
        ("static",      0.0,  1.0,  "transform_only"),
        ("uncertainty", None, None, "uncertainty"),
        ("gradnorm",    None, None, "gradnorm"),
    ]

    ablation_results = []

    print(f"\n{'#'*60}")
    print("ABLATION STUDY: Loss Weight Sweep")
    print(f"{'#'*60}")

    for method, w1, w2, label in weight_configs:
        print(f"\n--- Config: method={method}, label={label} ---")

        if method == "static":
            loss_weighter = StaticWeighter(w1, w2)
        elif method == "uncertainty":
            loss_weighter = UncertaintyLoss()
        elif method == "gradnorm":
            loss_weighter = GradNormBalancer(CONFIG["gradnorm_alpha"], CONFIG["gradnorm_weight_lr"])

        model = MultiTaskModel(backbone_name=CONFIG["backbone"]).to(CONFIG["device"])
        model, logger = train_multitask_model(
            model, train_loader, val_loader, CONFIG,
            loss_weighter=loss_weighter, tag=f"ablation_{label}"
        )

        criterion_bin = nn.CrossEntropyLoss()
        criterion_trans = nn.CrossEntropyLoss()
        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, CONFIG["device"])

        result = {
            "method": method,
            "w1": w1,
            "w2": w2,
            "label": label,
            "val_acc_bin": val_metrics["acc_bin"],
            "val_acc_trans": val_metrics["acc_trans"],
        }
        ablation_results.append(result)
        print(f"  Final Val -> Real/Fake: {val_metrics['acc_bin']*100:.2f}%, "
              f"Transform: {val_metrics['acc_trans']*100:.2f}%")

    ablation_df = pd.DataFrame(ablation_results)

    # --- Plot Pareto frontier ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ablation_df["val_acc_bin"] * 100, ablation_df["val_acc_trans"] * 100,
            'o-', markersize=10, linewidth=2, color="#4C72B0")

    for _, row in ablation_df.iterrows():
        ax.annotate(
            row["label"],
            (row["val_acc_bin"] * 100, row["val_acc_trans"] * 100),
            textcoords="offset points", xytext=(10, 10), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
        )

    ax.set_xlabel("Real/Fake Validation Accuracy (%)", fontsize=12)
    ax.set_ylabel("Transformation Validation Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study: Task Accuracy Trade-off (Pareto Frontier)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_pareto.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # --- Bar chart comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ablation_df))
    width = 0.35

    ax.bar(x - width/2, ablation_df["val_acc_bin"] * 100, width, label="Real/Fake Acc", color="#4C72B0")
    ax.bar(x + width/2, ablation_df["val_acc_trans"] * 100, width, label="Transform Acc", color="#DD8452")

    ax.set_xlabel("Weight Configuration")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Ablation Study: Effect of Loss Weights on Task Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["label"].tolist())
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_bars.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return ablation_df
