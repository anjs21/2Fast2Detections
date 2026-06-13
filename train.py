import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import *
from data import *
from globals import *
from network import *
from evaluation import *

def train_multitask_epoch(model, loader, optimizer, criterion_bin, criterion_trans,
                           device, w1=0.5, w2=0.5, grad_clip_norm=1.0,
                           scaler=None, use_amp=False):
    model.train()
    running_loss, running_loss_bin, running_loss_trans = 0.0, 0.0, 0.0
    correct_bin, correct_trans, total = 0, 0, 0

    for images, labels_bin, labels_trans in loader:
        images = images.to(device, non_blocking=True)
        labels_bin = labels_bin.to(device, non_blocking=True)
        labels_trans = labels_trans.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
            out_bin, out_trans = model(images)
            loss_bin = criterion_bin(out_bin, labels_bin)
            loss_trans = criterion_trans(out_trans, labels_trans)
            total_loss = w1 * loss_bin + w2 * loss_trans

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = images.size(0)
        running_loss += total_loss.item() * bs
        running_loss_bin += loss_bin.item() * bs
        running_loss_trans += loss_trans.item() * bs
        correct_bin += (torch.max(out_bin, 1)[1] == labels_bin).sum().item()
        correct_trans += (torch.max(out_trans, 1)[1] == labels_trans).sum().item()
        total += bs

    return {
        "loss": running_loss / total,
        "loss_bin": running_loss_bin / total,
        "loss_trans": running_loss_trans / total,
        "acc_bin": correct_bin / total,
        "acc_trans": correct_trans / total,
    }

@torch.no_grad()
def evaluate_multitask(model, loader, criterion_bin, criterion_trans, device):
    model.eval()
    running_loss_bin, running_loss_trans = 0.0, 0.0
    correct_bin, correct_trans, total = 0, 0, 0

    for images, labels_bin, labels_trans in loader:
        images = images.to(device, non_blocking=True)
        labels_bin = labels_bin.to(device, non_blocking=True)
        labels_trans = labels_trans.to(device, non_blocking=True)

        out_bin, out_trans = model(images)
        running_loss_bin += criterion_bin(out_bin, labels_bin).item() * images.size(0)
        running_loss_trans += criterion_trans(out_trans, labels_trans).item() * images.size(0)

        correct_bin += (torch.max(out_bin, 1)[1] == labels_bin).sum().item()
        correct_trans += (torch.max(out_trans, 1)[1] == labels_trans).sum().item()
        total += images.size(0)

    return {
        "loss_bin": running_loss_bin / total,
        "loss_trans": running_loss_trans / total,
        "acc_bin": correct_bin / total,
        "acc_trans": correct_trans / total,
    }

def train_singletask_epoch(model, loader, optimizer, criterion, device,
                            grad_clip_norm=1.0, scaler=None, use_amp=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
            out = model(images)
            loss = criterion(out, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        correct += (torch.max(out, 1)[1] == labels).sum().item()
        total += bs

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate_singletask(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(images)
        running_loss += criterion(out, labels).item() * images.size(0)
        correct += (torch.max(out, 1)[1] == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total

def train_multitask_model(model, train_loader, val_loader, config, w1=0.5, w2=0.5, tag="multitask"):
    device = config["device"]
    use_amp = config.get("amp", False) and device == "cuda"

    train_df = train_loader.dataset.df
    w_bin = compute_class_weights(train_df, "binary_label", BINARY_MAP, device)
    w_trans = compute_class_weights(train_df, "transform_label", TRANSFORM_MAP, device)
    criterion_bin = nn.CrossEntropyLoss(weight=w_bin)
    criterion_trans = nn.CrossEntropyLoss(weight=w_trans)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_score = 0.0
    best_model_state = None

    print(f"\nTraining [{tag}] | w1={w1}, w2={w2} | {config['epochs']} epochs")
    for epoch in range(config["epochs"]):
        train_metrics = train_multitask_epoch(
            model, train_loader, optimizer, criterion_bin, criterion_trans,
            device, w1, w2, config["grad_clip_norm"],
            scaler=scaler, use_amp=use_amp,
        )
        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, device)
        scheduler.step()

        val_score = (val_metrics["acc_bin"] + val_metrics["acc_trans"]) / 2

        logger.log(
            epoch + 1,
            train_loss=train_metrics["loss"],
            train_loss_bin=train_metrics["loss_bin"],
            train_loss_trans=train_metrics["loss_trans"],
            train_acc_bin=train_metrics["acc_bin"],
            train_acc_trans=train_metrics["acc_trans"],
            val_loss_bin=val_metrics["loss_bin"],
            val_loss_trans=val_metrics["loss_trans"],
            val_acc_bin=val_metrics["acc_bin"],
            val_acc_trans=val_metrics["acc_trans"],
            lr=optimizer.param_groups[0]["lr"],
        )

        print(f"Epoch [{epoch+1}/{config['epochs']}] Loss: {train_metrics['loss']:.4f}")
        print(f"  [Train] Real/Fake: {train_metrics['acc_bin']*100:.2f}% | Transform: {train_metrics['acc_trans']*100:.2f}%")
        print(f"  [Val]   Real/Fake: {val_metrics['acc_bin']*100:.2f}% | Transform: {val_metrics['acc_trans']*100:.2f}%")

        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = deepcopy(model.state_dict())
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{tag}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"  ★ New best model saved ({val_score*100:.2f}%)")

        early_stop(val_score)
        if early_stop.should_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, logger

def train_singletask_model(model, train_loader, val_loader, config, task_name="binary"):
    device = config["device"]
    use_amp = config.get("amp", False) and device == "cuda"

    train_df = train_loader.dataset.df
    if task_name == "binary":
        weights = compute_class_weights(train_df, "binary_label", BINARY_MAP, device)
    else:
        weights = compute_class_weights(train_df, "transform_label", TRANSFORM_MAP, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_acc = 0.0
    best_model_state = None

    print(f"\nTraining Unimodal Baseline [{task_name}] | {config['epochs']} epochs")
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_singletask_epoch(
            model, train_loader, optimizer, criterion, device,
            config["grad_clip_norm"], scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_acc = evaluate_singletask(model, val_loader, criterion, device)
        scheduler.step()

        logger.log(
            epoch + 1,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc,
            lr=optimizer.param_groups[0]["lr"],
        )

        print(f"Epoch [{epoch+1}/{config['epochs']}] Loss: {train_loss:.4f}")
        print(f"  [Train] Acc: {train_acc*100:.2f}% | [Val] Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_unimodal_{task_name}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"  ★ New best model saved ({val_acc*100:.2f}%)")

        early_stop(val_acc)
        if early_stop.should_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, logger

def _build_singletask_model(num_classes, head_cfg):
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

def _build_multitask_model():
    if CONFIG.get("dual_stream"):
        return DualStreamMultiTaskModel(
            rgb_backbone_name=CONFIG["backbone"],
            noise_backbone_name=CONFIG.get("noise_backbone", "resnet18"),
            trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
            dropout=CONFIG["dropout"],
            binary_head_cfg=CONFIG.get("binary_head"),
            transform_head_cfg=CONFIG.get("transform_head"),
        ).to(CONFIG["device"])
    return MultiTaskModel(
        backbone_name=CONFIG["backbone"],
        trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
        dropout=CONFIG["dropout"],
        binary_head_cfg=CONFIG.get("binary_head"),
        transform_head_cfg=CONFIG.get("transform_head"),
    ).to(CONFIG["device"])

def run_unimodal_baselines(st_bin_train, st_bin_val, st_bin_test,
                           st_trans_train, st_trans_val, st_trans_test):
    print("\n" + "=" * 60)
    print("PHASE 2: UNIMODAL BASELINES")
    print("=" * 60)

    if CONFIG.get("dual_stream"):
        print(f"[model] Unimodal baselines use dual-stream: RGB={CONFIG['backbone']} + Bayar noise stream={CONFIG.get('noise_backbone', 'resnet18')}")

    model_bin_only = _build_singletask_model(num_classes=2, head_cfg=CONFIG.get("binary_head"))
    model_bin_only, logger_bin = train_singletask_model(
        model_bin_only, st_bin_train, st_bin_val, CONFIG, task_name="binary"
    )
    plot_training_curves(logger_bin, "Unimodal Baseline: Binary (Real/Fake)", "curves_unimodal_binary.png")

    preds_bin_uni, labels_bin_uni = collect_singletask_predictions(
        model_bin_only, st_bin_test, CONFIG["device"]
    )
    unimodal_bin_acc = (preds_bin_uni == labels_bin_uni).mean()
    full_classification_report(preds_bin_uni, labels_bin_uni, ["Real", "Fake"], "Unimodal Binary Baseline (TEST)")

    model_trans_only = _build_singletask_model(num_classes=3, head_cfg=CONFIG.get("transform_head"))
    model_trans_only, logger_trans = train_singletask_model(
        model_trans_only, st_trans_train, st_trans_val, CONFIG, task_name="transform"
    )
    plot_training_curves(logger_trans, "Unimodal Baseline: Transformation Type", "curves_unimodal_transform.png")

    preds_trans_uni, labels_trans_uni = collect_singletask_predictions(
        model_trans_only, st_trans_test, CONFIG["device"]
    )
    unimodal_trans_acc = (preds_trans_uni == labels_trans_uni).mean()
    full_classification_report(preds_trans_uni, labels_trans_uni, ["Original", "Transmitted", "Redigitalized"], "Unimodal Transform Baseline (TEST)")

    return unimodal_bin_acc, unimodal_trans_acc

def run_multimodal_training(mt_train_loader, mt_val_loader, mt_test_loader, test_df, val_tfm):
    print("\n" + "=" * 60)
    print("PHASE 3: MULTI-TASK JOINT TRAINING")
    print("=" * 60)

    if CONFIG.get("dual_stream"):
        print(f"[model] Dual-stream: RGB={CONFIG['backbone']} + Bayar noise stream={CONFIG.get('noise_backbone', 'resnet18')}")

    model_mt = _build_multitask_model()
    model_mt, logger_mt = train_multitask_model(
        model_mt, mt_train_loader, mt_val_loader, CONFIG, w1=CONFIG["binary_weight"], w2=1-CONFIG["binary_weight"], tag="multitask_equal"
    )
    plot_training_curves(logger_mt, "Multi-Task Joint Training (w1={:.2f}, w2={:.2f})".format(CONFIG["binary_weight"], 1-CONFIG["binary_weight"]), "curves_multitask.png")

    results = collect_multitask_predictions(model_mt, mt_test_loader, CONFIG["device"])
    full_classification_report(results["preds_bin"], results["labels_bin"], ["Real", "Fake"], "Multi-Task Real/Fake (TEST)", probs=results["probs_bin"])
    full_classification_report(results["preds_trans"], results["labels_trans"], ["Original", "Transmitted", "Redigitalized"], "Multi-Task Transformation (TEST)")

    multitask_bin_acc = (results["preds_bin"] == results["labels_bin"]).mean()
    multitask_trans_acc = (results["preds_trans"] == results["labels_trans"]).mean()

    return model_mt, results, multitask_bin_acc, multitask_trans_acc

def run_ablation_study(train_loader, val_loader):
    weight_configs = [
        (1.0, 0.0, "binary_only"),
        (0.75, 0.25, "binary_dominant"),
        (0.5, 0.5, "equal"),
        (0.25, 0.75, "transform_dominant"),
        (0.0, 1.0, "transform_only"),
    ]

    ablation_results = []
    print("\nABLATION STUDY: Loss Weight Sweep")

    for w1, w2, label in weight_configs:
        print(f"\n--- Config: w1={w1}, w2={w2} ({label}) ---")
        model = _build_multitask_model()
        model, logger = train_multitask_model(
            model, train_loader, val_loader, CONFIG, w1=w1, w2=w2, tag=f"ablation_{label}"
        )

        criterion_bin = nn.CrossEntropyLoss()
        criterion_trans = nn.CrossEntropyLoss()
        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, CONFIG["device"])

        result = {
            "w1": w1,
            "w2": w2,
            "label": label,
            "val_acc_bin": val_metrics["acc_bin"],
            "val_acc_trans": val_metrics["acc_trans"],
        }
        ablation_results.append(result)
        print(f"  Final Val -> Real/Fake: {val_metrics['acc_bin']*100:.2f}%, Transform: {val_metrics['acc_trans']*100:.2f}%")

    ablation_df = pd.DataFrame(ablation_results)

    # Plot Pareto Frontier
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ablation_df["val_acc_bin"] * 100, ablation_df["val_acc_trans"] * 100, 'o-', markersize=10, linewidth=2, color="#4C72B0")

    for _, row in ablation_df.iterrows():
        ax.annotate(
            f"w1={row['w1']:.2f}\nw2={row['w2']:.2f}",
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
    plt.close()

    # Plot Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ablation_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, ablation_df["val_acc_bin"] * 100, width, label="Real/Fake Acc", color="#4C72B0")
    bars2 = ax.bar(x + width/2, ablation_df["val_acc_trans"] * 100, width, label="Transform Acc", color="#DD8452")

    ax.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)

    ax.set_xlabel("Weight Configuration")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Ablation Study: Effect of Loss Weights on Task Performance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"w1={r['w1']}\nw2={r['w2']}" for _, r in ablation_df.iterrows()])
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return ablation_df

if __name__ == "__main__":
    start_time = time.time()
    print(f"Training with {CONFIG['backbone']} backbone with {CONFIG['trainable_backbone_stages']} trainable backbone stages")
    print(f"[seed] CONFIG seed = {CONFIG['seed']}")

    # 1. Download & Prepare Splits
    download_train_val_data()
    download_test_data()

    train_df, val_df, test_df = build_splits(
        ORIGINAL_TRAIN_DIR, ORIGINAL_VAL_DIR, TEST_SUBSET_DIR,
        subset_per_class=CONFIG["subset_per_class"], seed=CONFIG["seed"],
        val_frac=CONFIG["val_frac"], test_frac=CONFIG["test_frac"],
        csv_path=METADATA_TRAIN_VAL_CSV,
    )

    print(f"\nTrain subset size: {len(train_df)}")
    print(f"Validation subset size: {len(val_df)}")
    print(f"Test subset size: {len(test_df)}")

    # 2. Transforms & Loaders
    train_tfm = get_train_transform(CONFIG["img_size"], CONFIG["backbone"])
    val_tfm = get_val_transform(CONFIG["img_size"], CONFIG["backbone"])
    test_tfm = get_tta_transform(CONFIG["img_size"], CONFIG["backbone"], CONFIG["tta_crops"]) if CONFIG.get("tta_crops", 1) > 1 else val_tfm

    def _loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=shuffle,
                          num_workers=CONFIG["num_workers"], pin_memory=True)

    mt_train_loader = _loader(MultiTaskDataset(train_df, transform=train_tfm), True)
    mt_val_loader = _loader(MultiTaskDataset(val_df, transform=val_tfm), False)
    mt_test_loader = _loader(MultiTaskDataset(test_df, transform=test_tfm), False)

    st_bin_train = _loader(SingleTaskDataset(train_df, "binary", train_tfm), True)
    st_bin_val = _loader(SingleTaskDataset(val_df, "binary", val_tfm), False)
    st_bin_test = _loader(SingleTaskDataset(test_df, "binary", test_tfm), False)
    st_trans_train = _loader(SingleTaskDataset(train_df, "transform", train_tfm), True)
    st_trans_val = _loader(SingleTaskDataset(val_df, "transform", val_tfm), False)
    st_trans_test = _loader(SingleTaskDataset(test_df, "transform", test_tfm), False)

    # 3. Train Baselines
    unimodal_bin_acc, unimodal_trans_acc = run_unimodal_baselines(
        st_bin_train, st_bin_val, st_bin_test,
        st_trans_train, st_trans_val, st_trans_test
    )

    # 4. Train Multitask
    model_mt, results, multitask_bin_acc, multitask_trans_acc = run_multimodal_training(
        mt_train_loader, mt_val_loader, mt_test_loader, test_df, val_tfm
    )

    # 5. Analysis & Comparison
    comparison_df, breakdown_df, trace_df = run_comparison_and_analysis(
        results, test_df, val_tfm, model_mt,
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc
    )

    # 6. Ablation Sweep
    ablation_df = run_ablation_study(mt_train_loader, mt_val_loader)

    # 7. Save Outputs
    save_results(
        model_mt, start_time,
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc,
        ablation_df, breakdown_df, trace_df
    )
