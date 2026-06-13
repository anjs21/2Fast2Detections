"""
Evaluation, analysis, and visualization functions.
"""

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from PIL import Image

from config import (
    RESULTS_DIR, BINARY_MAP, INV_BINARY_MAP, INV_TRANSFORM_MAP,
)


# =============================================================================
# Prediction Collection
# =============================================================================
@torch.no_grad()
def collect_multitask_predictions(model, loader, device):
    """Collect all predictions and labels from a multi-task model.

    Supports multi-crop TTA batches: if images arrive as [B, K, C, H, W]
    (see data.get_tta_transform), the K crops are run in one forward and the
    softmax is averaged over crops before the argmax.
    """
    model.eval()
    all_preds_bin, all_labels_bin = [], []
    all_preds_trans, all_labels_trans = [], []
    all_probs_bin = []

    for images, labels_bin, labels_trans in loader:
        images = images.to(device)
        if images.dim() == 5:  # [B, K, C, H, W] TTA crops
            b, k = images.shape[:2]
            out_bin, out_trans = model(images.flatten(0, 1))
            probs_bin = torch.softmax(out_bin, dim=1).view(b, k, -1).mean(dim=1)
            probs_trans = torch.softmax(out_trans, dim=1).view(b, k, -1).mean(dim=1)
        else:
            out_bin, out_trans = model(images)
            probs_bin = torch.softmax(out_bin, dim=1)
            probs_trans = torch.softmax(out_trans, dim=1)

        all_probs_bin.extend(probs_bin.cpu().numpy())

        all_preds_bin.extend(torch.max(probs_bin, 1)[1].cpu().numpy())
        all_labels_bin.extend(labels_bin.numpy())

        all_preds_trans.extend(torch.max(probs_trans, 1)[1].cpu().numpy())
        all_labels_trans.extend(labels_trans.numpy())

    return {
        "preds_bin": np.array(all_preds_bin),
        "labels_bin": np.array(all_labels_bin),
        "probs_bin": np.array(all_probs_bin),
        "preds_trans": np.array(all_preds_trans),
        "labels_trans": np.array(all_labels_trans),
    }


@torch.no_grad()
def collect_singletask_predictions(model, loader, device):
    """Collect all predictions and labels from a single-task model.

    Supports multi-crop TTA batches ([B, K, C, H, W]) by averaging the softmax
    over the K crops, as in collect_multitask_predictions.
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        if images.dim() == 5:  # [B, K, C, H, W] TTA crops
            b, k = images.shape[:2]
            out = model(images.flatten(0, 1))
            probs = torch.softmax(out, dim=1).view(b, k, -1).mean(dim=1)
        else:
            probs = torch.softmax(model(images), dim=1)
        all_preds.extend(torch.max(probs, 1)[1].cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# =============================================================================
# Classification Reports & Confusion Matrices
# =============================================================================
def full_classification_report(preds, labels, class_names, task_title, probs=None):
    """Print a full classification report with confusion matrix.

    If `probs` (softmax outputs) is given for a binary task, also reports ROC-AUC
    and average precision — the standard detection metrics in the AI-gen
    detection literature (e.g. the RRBench benchmark)."""
    print(f"\n{'='*25} {task_title} {'='*25}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    if probs is not None and len(class_names) == 2:
        probs = np.asarray(probs)
        scores = probs[:, 1] if probs.ndim == 2 else probs  # P(fake)
        try:
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            print(f"  ROC-AUC: {auc:.4f} | Average Precision: {ap:.4f}")
        except ValueError as e:
            print(f"  (AUC unavailable: {e})")

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"{task_title} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"cm_{task_title.replace(' ', '_').replace('/', '_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    return cm


# =============================================================================
# Per-Transformation Breakdown
# =============================================================================
def per_transformation_breakdown(results, val_df):
    """
    Break down real/fake detection accuracy separately for each
    transformation category (original, transmitted, redigitalized).
    Also check whether the pattern differs between real and AI-generated images.

    NOTE: the brief asks for a breakdown by "each re-digitization method"
    (scan / printout-photo / screen-photo / projection). The extracted
    test_subset filenames encode only the scenario (e.g. "redigital_normal_..."),
    not the re-digitization method, so a per-method split is not possible with
    this subset — the breakdown is at the transform-category level.
    """
    val_df_reset = val_df.reset_index(drop=True)

    preds_bin = results["preds_bin"]
    labels_bin = results["labels_bin"]
    labels_trans = results["labels_trans"]

    print(f"\n{'='*60}")
    print("PER-TRANSFORMATION REAL/FAKE ACCURACY BREAKDOWN")
    print(f"{'='*60}")

    rows = []
    for trans_id, trans_name in INV_TRANSFORM_MAP.items():
        mask = labels_trans == trans_id
        if mask.sum() == 0:
            continue

        subset_preds = preds_bin[mask]
        subset_labels = labels_bin[mask]

        # Overall accuracy for this transformation
        overall_acc = (subset_preds == subset_labels).mean()

        # Accuracy for real images under this transformation
        real_mask = subset_labels == BINARY_MAP["real"]
        real_acc = (subset_preds[real_mask] == subset_labels[real_mask]).mean() if real_mask.sum() > 0 else 0.0

        # Accuracy for fake images under this transformation
        fake_mask = subset_labels == BINARY_MAP["fake"]
        fake_acc = (subset_preds[fake_mask] == subset_labels[fake_mask]).mean() if fake_mask.sum() > 0 else 0.0

        n_total = mask.sum()
        n_real = real_mask.sum()
        n_fake = fake_mask.sum()

        print(f"\n  [{trans_name.upper()}] (n={n_total})")
        print(f"    Overall Real/Fake Acc: {overall_acc*100:.2f}%")
        print(f"    Real images correct:   {real_acc*100:.2f}% (n={n_real})")
        print(f"    Fake images correct:   {fake_acc*100:.2f}% (n={n_fake})")

        rows.append({
            "transformation": trans_name,
            "overall_acc": overall_acc,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "n_total": int(n_total),
            "n_real": int(n_real),
            "n_fake": int(n_fake),
        })

    breakdown_df = pd.DataFrame(rows)

    # --- Plot grouped bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(breakdown_df))
    width = 0.25

    bars1 = ax.bar(x - width, breakdown_df["overall_acc"] * 100, width, label="Overall", color="#4C72B0")
    bars2 = ax.bar(x, breakdown_df["real_acc"] * 100, width, label="Real images", color="#55A868")
    bars3 = ax.bar(x + width, breakdown_df["fake_acc"] * 100, width, label="Fake images", color="#C44E52")

    ax.set_xlabel("Transformation Type")
    ax.set_ylabel("Real/Fake Detection Accuracy (%)")
    ax.set_title("Real/Fake Accuracy Breakdown by Transformation Type")
    ax.set_xticks(x)
    ax.set_xticklabels(breakdown_df["transformation"].str.capitalize())
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_transformation_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return breakdown_df


# =============================================================================
# Cross-Class Trace Analysis
# =============================================================================
def cross_class_trace_analysis(results):
    """
    Investigate whether AI-generated and real images respond differently
    to the same post-processing operations.
    """
    preds_bin = results["preds_bin"]
    labels_bin = results["labels_bin"]
    preds_trans = results["preds_trans"]
    labels_trans = results["labels_trans"]
    probs_bin = results["probs_bin"]

    print(f"\n{'='*60}")
    print("CROSS-CLASS TRANSFORMATION TRACE ANALYSIS")
    print(f"{'='*60}")

    # Build a 6-cell analysis: (real/fake) x (original/transmitted/redigitalized)
    rows = []
    for bin_id, bin_name in INV_BINARY_MAP.items():
        for trans_id, trans_name in INV_TRANSFORM_MAP.items():
            mask = (labels_bin == bin_id) & (labels_trans == trans_id)
            if mask.sum() == 0:
                continue

            subset_preds_bin = preds_bin[mask]
            subset_labels_bin = labels_bin[mask]
            subset_probs = probs_bin[mask]

            bin_acc = (subset_preds_bin == subset_labels_bin).mean()
            avg_confidence = subset_probs[np.arange(len(subset_labels_bin)), subset_labels_bin].mean()
            misclass_rate = 1.0 - bin_acc

            # Transformation classification accuracy for this cell
            subset_preds_trans = preds_trans[mask]
            subset_labels_trans = labels_trans[mask]
            trans_acc = (subset_preds_trans == subset_labels_trans).mean()

            rows.append({
                "binary_class": bin_name,
                "transform_type": trans_name,
                "n_samples": int(mask.sum()),
                "binary_acc": bin_acc,
                "binary_misclass_rate": misclass_rate,
                "avg_confidence": avg_confidence,
                "transform_acc": trans_acc,
            })

    trace_df = pd.DataFrame(rows)
    print("\nDetailed 6-cell analysis (binary_class × transform_type):")
    print(trace_df.to_string(index=False, float_format="%.4f"))

    # --- Heatmap: Binary accuracy across the 2x3 grid ---
    pivot_acc = trace_df.pivot(index="binary_class", columns="transform_type", values="binary_acc")
    pivot_misclass = trace_df.pivot(index="binary_class", columns="transform_type", values="binary_misclass_rate")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(pivot_acc * 100, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0],
                vmin=50, vmax=100)
    axes[0].set_title("Real/Fake Accuracy (%) by Class × Transformation")
    axes[0].set_ylabel("True Binary Class")
    axes[0].set_xlabel("Transformation Type")

    sns.heatmap(pivot_misclass * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1],
                vmin=0, vmax=50)
    axes[1].set_title("Misclassification Rate (%) by Class × Transformation")
    axes[1].set_ylabel("True Binary Class")
    axes[1].set_xlabel("Transformation Type")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_class_traces.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # --- Confidence distribution comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (trans_id, trans_name) in enumerate(INV_TRANSFORM_MAP.items()):
        ax = axes[i]
        for bin_id, bin_name in INV_BINARY_MAP.items():
            mask = (labels_bin == bin_id) & (labels_trans == trans_id)
            if mask.sum() == 0:
                continue
            # Confidence of the correct class
            correct_conf = probs_bin[mask, bin_id]
            ax.hist(correct_conf, bins=20, alpha=0.5, label=f"{bin_name.capitalize()} images",
                    density=True, edgecolor="black", linewidth=0.5)

        ax.set_title(f"Confidence Distribution\n({trans_name.capitalize()})")
        ax.set_xlabel("P(correct class)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_distributions.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return trace_df


# =============================================================================
# Training Curve Visualization
# =============================================================================
def plot_training_curves(logger, title="Training Curves", filename="training_curves.png"):
    """Plot loss and accuracy curves from a TrainingLogger."""
    history = logger.to_dataframe()

    # Detect if this is a multi-task or single-task logger
    is_multitask = "train_acc_bin" in history.columns

    if is_multitask:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(history["epoch"], history["train_loss"], 'o-', label="Train Total Loss")
        axes[0, 0].plot(history["epoch"], history["train_loss_bin"], 's--', label="Train Binary Loss", alpha=0.7)
        axes[0, 0].plot(history["epoch"], history["train_loss_trans"], '^--', label="Train Transform Loss", alpha=0.7)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Val Loss
        axes[0, 1].plot(history["epoch"], history["val_loss_bin"], 's-', label="Val Binary Loss", color="orange")
        axes[0, 1].plot(history["epoch"], history["val_loss_trans"], '^-', label="Val Transform Loss", color="red")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Train Accuracy
        axes[1, 0].plot(history["epoch"], history["train_acc_bin"].apply(lambda x: x*100), 'o-', label="Binary")
        axes[1, 0].plot(history["epoch"], history["train_acc_trans"].apply(lambda x: x*100), 's-', label="Transform")
        axes[1, 0].set_title("Training Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Val Accuracy
        axes[1, 1].plot(history["epoch"], history["val_acc_bin"].apply(lambda x: x*100), 'o-', label="Binary")
        axes[1, 1].plot(history["epoch"], history["val_acc_trans"].apply(lambda x: x*100), 's-', label="Transform")
        axes[1, 1].set_title("Validation Accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(history["epoch"], history["train_loss"], 'o-', label="Train")
        axes[0].plot(history["epoch"], history["val_loss"], 's-', label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["epoch"], history["train_acc"].apply(lambda x: x*100), 'o-', label="Train")
        axes[1].plot(history["epoch"], history["val_acc"].apply(lambda x: x*100), 's-', label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Visual Inference
# =============================================================================
def run_visual_inference(model, df, val_transform, device, num_samples=6):
    """Display model predictions on random images."""
    model.eval()
    samples = df.sample(num_samples, random_state=np.random.randint(1, 10000))

    cols = min(num_samples, 3)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(samples.iterrows()):
        raw_img = Image.open(row["filepath"]).convert("RGB")
        input_tensor = val_transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_bin, out_trans = model(input_tensor)
            p_bin = INV_BINARY_MAP[torch.max(out_bin, 1)[1].item()]
            p_trans = INV_TRANSFORM_MAP[torch.max(out_trans, 1)[1].item()]

        ax = axes[i]
        ax.imshow(raw_img)
        is_correct = (p_bin == row["binary_label"]) and (p_trans == row["transform_label"])
        title_color = "green" if is_correct else "red"
        ax.set_title(
            f"True: {row['binary_label']} | {row['transform_label']}\n"
            f"Pred: {p_bin} | {p_trans}",
            color=title_color, fontsize=11
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "visual_inference.png"), dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Comparative Summary
# =============================================================================
def print_comparative_summary(unimodal_bin_acc, unimodal_trans_acc,
                              multitask_bin_acc, multitask_trans_acc):
    """Compare unimodal baselines against multi-task model."""
    print(f"\n{'#'*60}")
    print("UNIMODAL vs MULTI-TASK COMPARISON")
    print(f"{'#'*60}")

    data = {
        "Model": ["Unimodal (Binary only)", "Unimodal (Transform only)", "Multi-task (Joint)"],
        "Real/Fake Acc (%)": [
            f"{unimodal_bin_acc*100:.2f}",
            "N/A",
            f"{multitask_bin_acc*100:.2f}",
        ],
        "Transform Acc (%)": [
            "N/A",
            f"{unimodal_trans_acc*100:.2f}",
            f"{multitask_trans_acc*100:.2f}",
        ],
    }

    comparison_df = pd.DataFrame(data)
    print(comparison_df.to_string(index=False))

    # Determine if joint training helps
    bin_delta = multitask_bin_acc - unimodal_bin_acc
    trans_delta = multitask_trans_acc - unimodal_trans_acc

    print(f"\n  Delta (Binary):    {bin_delta*100:+.2f}% "
          f"{'↑ improved' if bin_delta > 0 else '↓ degraded' if bin_delta < 0 else '= unchanged'}")
    print(f"  Delta (Transform): {trans_delta*100:+.2f}% "
          f"{'↑ improved' if trans_delta > 0 else '↓ degraded' if trans_delta < 0 else '= unchanged'}")

    if bin_delta > 0 and trans_delta > 0:
        print("\n  ✓ Joint training IMPROVES both tasks — the tasks complement each other.")
    elif bin_delta < 0 and trans_delta < 0:
        print("\n  ✗ Joint training DEGRADES both tasks — the tasks compete for capacity.")
    else:
        print("\n  ~ Joint training has mixed effects — partial complementarity.")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    models_list = ["Unimodal\n(Binary)", "Unimodal\n(Transform)", "Multi-task\n(Joint)"]
    bin_accs = [unimodal_bin_acc * 100, 0, multitask_bin_acc * 100]
    trans_accs = [0, unimodal_trans_acc * 100, multitask_trans_acc * 100]

    x = np.arange(len(models_list))
    width = 0.35

    bars1 = ax.bar(x - width/2, bin_accs, width, label="Real/Fake Acc", color="#4C72B0")
    bars2 = ax.bar(x + width/2, trans_accs, width, label="Transform Acc", color="#DD8452")

    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Unimodal Baselines vs Multi-Task Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "unimodal_vs_multitask.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return comparison_df
