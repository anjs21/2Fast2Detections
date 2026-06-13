import os
import json
import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from PIL import Image
from globals import *

@torch.no_grad()
def collect_multitask_predictions(model, loader, device):
    """Collect predictions and labels from a multi-task model (TTA supported)."""
    model.eval()
    all_preds_bin, all_labels_bin = [], []
    all_preds_trans, all_labels_trans = [], []
    all_probs_bin = []

    for images, labels_bin, labels_trans in loader:
        images = images.to(device)
        if images.dim() == 5:  # TTA mode [B, K, C, H, W]
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
    """Collect predictions and labels from a single-task model (TTA supported)."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        if images.dim() == 5:
            b, k = images.shape[:2]
            out = model(images.flatten(0, 1))
            probs = torch.softmax(out, dim=1).view(b, k, -1).mean(dim=1)
        else:
            probs = torch.softmax(model(images), dim=1)
        all_preds.extend(torch.max(probs, 1)[1].cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

def full_classification_report(preds, labels, class_names, task_title, probs=None):
    """Generate and display classification reports and save confusion matrix heatmaps."""
    print(f"\n{'='*25} {task_title} {'='*25}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    if probs is not None and len(class_names) == 2:
        probs = np.asarray(probs)
        scores = probs[:, 1] if probs.ndim == 2 else probs
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
    plt.close()
    return cm

def per_transformation_breakdown(results, val_df):
    """Breakdown binary accuracy across transformation types."""
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

        overall_acc = (subset_preds == subset_labels).mean()
        real_mask = subset_labels == BINARY_MAP["real"]
        real_acc = (subset_preds[real_mask] == subset_labels[real_mask]).mean() if real_mask.sum() > 0 else 0.0
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

    ax.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=8)
    ax.bar_label(bars3, fmt='%.1f%%', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_transformation_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return breakdown_df

def cross_class_trace_analysis(results):
    """Analyze predictions across a 2x3 grid of class vs transformation."""
    preds_bin = results["preds_bin"]
    labels_bin = results["labels_bin"]
    preds_trans = results["preds_trans"]
    labels_trans = results["labels_trans"]
    probs_bin = results["probs_bin"]

    print(f"\n{'='*60}")
    print("CROSS-CLASS TRANSFORMATION TRACE ANALYSIS")
    print(f"{'='*60}")

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
    print("\nDetailed 6-cell analysis:")
    print(trace_df.to_string(index=False, float_format="%.4f"))

    pivot_acc = trace_df.pivot(index="binary_class", columns="transform_type", values="binary_acc")
    pivot_misclass = trace_df.pivot(index="binary_class", columns="transform_type", values="binary_misclass_rate")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(pivot_acc * 100, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0], vmin=50, vmax=100)
    axes[0].set_title("Real/Fake Accuracy (%) by Class × Transformation")
    axes[0].set_ylabel("True Binary Class")
    axes[0].set_xlabel("Transformation Type")

    sns.heatmap(pivot_misclass * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1], vmin=0, vmax=50)
    axes[1].set_title("Misclassification Rate (%) by Class × Transformation")
    axes[1].set_ylabel("True Binary Class")
    axes[1].set_xlabel("Transformation Type")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_class_traces.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (trans_id, trans_name) in enumerate(INV_TRANSFORM_MAP.items()):
        ax = axes[i]
        for bin_id, bin_name in INV_BINARY_MAP.items():
            mask = (labels_bin == bin_id) & (labels_trans == trans_id)
            if mask.sum() == 0:
                continue
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
    plt.close()
    return trace_df

def plot_training_curves(logger, title="Training Curves", filename="training_curves.png"):
    """Plot metrics across epochs."""
    history = logger.to_dataframe()
    is_multitask = "train_acc_bin" in history.columns

    if is_multitask:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(history["epoch"], history["train_loss"], 'o-', label="Train Total Loss")
        axes[0, 0].plot(history["epoch"], history["train_loss_bin"], 's--', label="Train Binary Loss", alpha=0.7)
        axes[0, 0].plot(history["epoch"], history["train_loss_trans"], '^--', label="Train Transform Loss", alpha=0.7)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history["epoch"], history["val_loss_bin"], 's-', label="Val Binary Loss", color="orange")
        axes[0, 1].plot(history["epoch"], history["val_loss_trans"], '^-', label="Val Transform Loss", color="red")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(history["epoch"], history["train_acc_bin"].apply(lambda x: x*100), 'o-', label="Binary")
        axes[1, 0].plot(history["epoch"], history["train_acc_trans"].apply(lambda x: x*100), 's-', label="Transform")
        axes[1, 0].set_title("Training Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

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
    plt.close()

def run_visual_inference(model, df, val_transform, device, num_samples=6):
    """Performs visual inference on sample images and saves display plots."""
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

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "visual_inference.png"), dpi=150, bbox_inches="tight")
    plt.close()

def print_comparative_summary(unimodal_bin_acc, unimodal_trans_acc,
                               multitask_bin_acc, multitask_trans_acc):
    """Compares baseline unimodal accuracies against multitask accuracy."""
    print(f"\n{'#'*60}")
    print("UNIMODAL vs MULTI-TASK COMPARISON")
    print(f"{'#'*60}")

    data = {
        "Model": ["Unimodal (Binary only)", "Unimodal (Transform only)", "Multi-task (Joint)"],
        "Real/Fake Acc (%)": [
            f"{unimodal_bin_acc*100:.2f}" if unimodal_bin_acc is not None else "N/A",
            "N/A",
            f"{multitask_bin_acc*100:.2f}" if multitask_bin_acc is not None else "N/A",
        ],
        "Transform Acc (%)": [
            "N/A",
            f"{unimodal_trans_acc*100:.2f}" if unimodal_trans_acc is not None else "N/A",
            f"{multitask_trans_acc*100:.2f}" if multitask_trans_acc is not None else "N/A",
        ],
    }

    comparison_df = pd.DataFrame(data)
    print(comparison_df.to_string(index=False))

    if unimodal_bin_acc is not None and unimodal_trans_acc is not None and multitask_bin_acc is not None and multitask_trans_acc is not None:
        bin_delta = multitask_bin_acc - unimodal_bin_acc
        trans_delta = multitask_trans_acc - unimodal_trans_acc
        print(f"\n  Delta (Binary):    {bin_delta*100:+.2f}% "
              f"{'↑ improved' if bin_delta > 0 else '↓ degraded' if bin_delta < 0 else '= unchanged'}")
        print(f"  Delta (Transform): {trans_delta*100:+.2f}% "
              f"{'↑ improved' if trans_delta > 0 else '↓ degraded' if trans_delta < 0 else '= unchanged'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    models_list = ["Unimodal\n(Binary)", "Unimodal\n(Transform)", "Multi-task\n(Joint)"]
    bin_accs = [unimodal_bin_acc * 100 if unimodal_bin_acc is not None else 0, 0, multitask_bin_acc * 100 if multitask_bin_acc is not None else 0]
    trans_accs = [0, unimodal_trans_acc * 100 if unimodal_trans_acc is not None else 0, multitask_trans_acc * 100 if multitask_trans_acc is not None else 0]

    x = np.arange(len(models_list))
    width = 0.35

    bars1 = ax.bar(x - width/2, bin_accs, width, label="Real/Fake Acc", color="#4C72B0")
    bars2 = ax.bar(x + width/2, trans_accs, width, label="Transform Acc", color="#DD8452")

    labels1 = [f'{val:.1f}%' if val > 0 else '' for val in bin_accs]
    labels2 = [f'{val:.1f}%' if val > 0 else '' for val in trans_accs]
    ax.bar_label(bars1, labels=labels1, padding=3, fontsize=9)
    ax.bar_label(bars2, labels=labels2, padding=3, fontsize=9)

    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Unimodal Baselines vs Multi-Task Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "unimodal_vs_multitask.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return comparison_df

def run_comparison_and_analysis(results, test_df, val_tfm, model_mt,
                                 unimodal_bin_acc, unimodal_trans_acc,
                                 multitask_bin_acc, multitask_trans_acc):
    """Executes full suite of evaluations and visual analysis."""
    print("\n" + "=" * 60)
    print("PHASE 4: UNIMODAL vs MULTI-TASK COMPARISON")
    print("=" * 60)
    comparison_df = print_comparative_summary(
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc
    )

    print("\n" + "=" * 60)
    print("PHASE 5: PER-TRANSFORMATION ACCURACY BREAKDOWN")
    print("=" * 60)
    breakdown_df = per_transformation_breakdown(results, test_df)

    print("\n" + "=" * 60)
    print("PHASE 6: CROSS-CLASS TRANSFORMATION TRACE ANALYSIS")
    print("=" * 60)
    trace_df = cross_class_trace_analysis(results)

    print("\n" + "=" * 60)
    print("PHASE 8: VISUAL INFERENCE")
    print("=" * 60)
    run_visual_inference(model_mt, test_df, val_tfm, CONFIG["device"], num_samples=6)

    return comparison_df, breakdown_df, trace_df

def save_results(model_mt, start_time,
                 unimodal_bin_acc, unimodal_trans_acc,
                 multitask_bin_acc, multitask_trans_acc,
                 ablation_df, breakdown_df, trace_df):
    """Saves all evaluation results to JSON and serializes the model checkpoint."""
    print("\n" + "=" * 60)
    print("PHASE 9: SAVING RESULTS")
    print("=" * 60)

    final_summary = {
        "config": {k: str(v) for k, v in CONFIG.items()},
        "evaluation_split": "test",
        "unimodal_binary_test_acc": float(unimodal_bin_acc) if unimodal_bin_acc is not None else 0.0,
        "unimodal_transform_test_acc": float(unimodal_trans_acc) if unimodal_trans_acc is not None else 0.0,
        "multitask_binary_test_acc": float(multitask_bin_acc) if multitask_bin_acc is not None else 0.0,
        "multitask_transform_test_acc": float(multitask_trans_acc) if multitask_trans_acc is not None else 0.0,
        "ablation_results": ablation_df.to_dict(orient="records") if ablation_df is not None else [],
        "per_transformation_breakdown": breakdown_df.to_dict(orient="records") if breakdown_df is not None else [],
        "cross_class_traces": trace_df.to_dict(orient="records") if trace_df is not None else [],
        "total_training_time_minutes": (time.time() - start_time) / 60,
    }

    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    final_model_path = os.path.join(CHECKPOINTS_DIR, "final_multitask_model.pth")
    torch.save(model_mt.state_dict(), final_model_path)

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE! Total time: {elapsed:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Model checkpoints: {CHECKPOINTS_DIR}/")
    print(f"Experiment summary: {summary_path}")
    print(f"{'='*60}")
