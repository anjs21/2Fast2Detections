"""
Training utilities: EarlyStopping, TrainingLogger, SupCon loss, train/eval epoch
functions, and full training pipelines for both multi-task and single-task models.

DRCT-ConvB additions:
  - SupConLoss: supervised contrastive loss on the projection head, applied to
    binary real/fake labels (the diffusion-reconstructed hard-fakes from
    drct_reconstruct.py are ordinary "fake" rows, so they participate here).
  - Mixed-precision (AMP) training (no-op on CPU).
  - Class-weighted cross-entropy to counter the mild class imbalance.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from copy import deepcopy
import pandas as pd

from config import CHECKPOINTS_DIR, BINARY_MAP, TRANSFORM_MAP


# =============================================================================
# Training Utilities
# =============================================================================
class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


class TrainingLogger:
    """Track metrics across epochs for visualization."""

    def __init__(self):
        self.history = defaultdict(list)

    def log(self, epoch, **kwargs):
        self.history["epoch"].append(epoch)
        for k, v in kwargs.items():
            self.history[k].append(v)

    def to_dataframe(self):
        return pd.DataFrame(self.history)


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020), single-view variant.

    Pulls together normalized embeddings sharing a label and pushes apart those
    with different labels — here on binary real/fake labels, which is the core
    of DRCT's contrastive training.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        b = features.shape[0]
        if b < 2:
            return features.sum() * 0.0  # no pairs in batch

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)        # [B,B] positives (incl. self)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()  # stability

        self_mask = torch.eye(b, device=device)
        mask = mask * (1.0 - self_mask)                              # drop self-pairs
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count.clamp(min=1)
        has_pos = pos_count > 0
        if has_pos.sum() == 0:
            return features.sum() * 0.0
        return -(mean_log_prob_pos[has_pos]).mean()


def compute_class_weights(df, col, label_map, device):
    """Inverse-frequency class weights for cross-entropy."""
    counts = torch.zeros(len(label_map), dtype=torch.float)
    vc = df[col].map(label_map).value_counts()
    for idx, n in vc.items():
        counts[int(idx)] = n
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(label_map) * counts)
    return weights.to(device)


# =============================================================================
# Multi-Task Epoch Functions
# =============================================================================
def train_multitask_epoch(model, loader, optimizer, criterion_bin, criterion_trans,
                          device, w1=0.5, w2=0.5, grad_clip_norm=1.0,
                          contrastive_criterion=None, contrastive_weight=0.0,
                          scaler=None, use_amp=False):
    """Train one epoch for the multi-task model (with optional DRCT contrastive loss)."""
    model.train()
    running_loss, running_loss_bin, running_loss_trans, running_loss_con = 0.0, 0.0, 0.0, 0.0
    correct_bin, correct_trans, total = 0, 0, 0
    use_con = contrastive_criterion is not None and contrastive_weight > 0

    for images, labels_bin, labels_trans in loader:
        images = images.to(device, non_blocking=True)
        labels_bin = labels_bin.to(device, non_blocking=True)
        labels_trans = labels_trans.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            enabled=use_amp):
            if use_con:
                out_bin, out_trans, proj = model(images, return_proj=True)
            else:
                out_bin, out_trans = model(images)

            loss_bin = criterion_bin(out_bin, labels_bin)
            loss_trans = criterion_trans(out_trans, labels_trans)
            total_loss = w1 * loss_bin + w2 * loss_trans
            loss_con = torch.tensor(0.0, device=device)
            if use_con:
                loss_con = contrastive_criterion(proj.float(), labels_bin)
                total_loss = total_loss + contrastive_weight * loss_con

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
        running_loss_con += float(loss_con) * bs
        correct_bin += (torch.max(out_bin, 1)[1] == labels_bin).sum().item()
        correct_trans += (torch.max(out_trans, 1)[1] == labels_trans).sum().item()
        total += bs

    return {
        "loss": running_loss / total,
        "loss_bin": running_loss_bin / total,
        "loss_trans": running_loss_trans / total,
        "loss_con": running_loss_con / total,
        "acc_bin": correct_bin / total,
        "acc_trans": correct_trans / total,
    }


@torch.no_grad()
def evaluate_multitask(model, loader, criterion_bin, criterion_trans, device):
    """Evaluate the multi-task model and return loss + accuracy."""
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


# =============================================================================
# Single-Task Epoch Functions
# =============================================================================
def train_singletask_epoch(model, loader, optimizer, criterion, device,
                           grad_clip_norm=1.0, scaler=None, use_amp=False):
    """Train one epoch for a single-task model."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            enabled=use_amp):
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
    """Evaluate a single-task model."""
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


# =============================================================================
# Full Multi-Task Training Pipeline
# =============================================================================
def train_multitask_model(model, train_loader, val_loader, config, w1=0.5, w2=0.5, tag="multitask"):
    """Full training loop with LR scheduler, early stopping, AMP, DRCT contrastive
    loss, class-weighted CE, and best-model checkpointing."""
    device = config["device"]
    use_amp = config.get("amp", False) and device == "cuda"

    # Class-weighted CE (inverse frequency) from the training split.
    train_df = train_loader.dataset.df
    w_bin = compute_class_weights(train_df, "binary_label", BINARY_MAP, device)
    w_trans = compute_class_weights(train_df, "transform_label", TRANSFORM_MAP, device)
    criterion_bin = nn.CrossEntropyLoss(weight=w_bin)
    criterion_trans = nn.CrossEntropyLoss(weight=w_trans)

    use_con = config.get("use_drct", False) and getattr(model, "use_projection", False)
    contrastive_criterion = SupConLoss(config.get("contrastive_temp", 0.1)) if use_con else None
    contrastive_weight = config.get("contrastive_weight", 0.0) if use_con else 0.0

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_score = 0.0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Training [{tag}] | w1={w1}, w2={w2} | DRCT contrastive={contrastive_weight} | {config['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(config["epochs"]):
        train_metrics = train_multitask_epoch(
            model, train_loader, optimizer, criterion_bin, criterion_trans,
            device, w1, w2, config["grad_clip_norm"],
            contrastive_criterion=contrastive_criterion,
            contrastive_weight=contrastive_weight, scaler=scaler, use_amp=use_amp,
        )
        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, device)
        scheduler.step()

        val_score = (val_metrics["acc_bin"] + val_metrics["acc_trans"]) / 2

        logger.log(
            epoch + 1,
            train_loss=train_metrics["loss"],
            train_loss_bin=train_metrics["loss_bin"],
            train_loss_trans=train_metrics["loss_trans"],
            train_loss_con=train_metrics["loss_con"],
            train_acc_bin=train_metrics["acc_bin"],
            train_acc_trans=train_metrics["acc_trans"],
            val_loss_bin=val_metrics["loss_bin"],
            val_loss_trans=val_metrics["loss_trans"],
            val_acc_bin=val_metrics["acc_bin"],
            val_acc_trans=val_metrics["acc_trans"],
            lr=optimizer.param_groups[0]["lr"],
        )

        print(f"\nEpoch [{epoch+1}/{config['epochs']}] "
              f"Loss: {train_metrics['loss']:.4f} (con: {train_metrics['loss_con']:.4f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  [Train] Real/Fake: {train_metrics['acc_bin']*100:.2f}% | "
              f"Transform: {train_metrics['acc_trans']*100:.2f}%")
        print(f"  [Val]   Real/Fake: {val_metrics['acc_bin']*100:.2f}% | "
              f"Transform: {val_metrics['acc_trans']*100:.2f}%")

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
        print(f"\n✓ Restored best model (val score: {best_val_score*100:.2f}%)")

    return model, logger


# =============================================================================
# Full Single-Task Training Pipeline
# =============================================================================
def train_singletask_model(model, train_loader, val_loader, config, task_name="binary"):
    """Full training loop for a single-task baseline (class-weighted CE + AMP)."""
    device = config["device"]
    use_amp = config.get("amp", False) and device == "cuda"

    train_df = train_loader.dataset.df
    if task_name == "binary":
        weights = compute_class_weights(train_df, "binary_label", BINARY_MAP, device)
    else:
        weights = compute_class_weights(train_df, "transform_label", TRANSFORM_MAP, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_acc = 0.0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Training Unimodal Baseline [{task_name}] | {config['epochs']} epochs")
    print(f"{'='*60}")

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

        print(f"\nEpoch [{epoch+1}/{config['epochs']}] "
              f"Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  [Train] Acc: {train_acc*100:.2f}%")
        print(f"  [Val]   Acc: {val_acc*100:.2f}%")

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
        print(f"\n✓ Restored best model (val acc: {best_val_acc*100:.2f}%)")

    return model, logger
