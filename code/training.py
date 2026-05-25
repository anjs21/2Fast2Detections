"""
Training utilities: EarlyStopping, TrainingLogger, train/eval epoch functions,
and full training pipelines for both multi-task and single-task models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from copy import deepcopy
import pandas as pd

from config import CHECKPOINTS_DIR


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


# =============================================================================
# Multi-Task Epoch Functions
# =============================================================================
def train_multitask_epoch(model, loader, optimizer, criterion_bin, criterion_trans,
                          device, w1=0.5, w2=0.5, grad_clip_norm=1.0):
    """Train one epoch for the multi-task model."""
    model.train()
    running_loss, running_loss_bin, running_loss_trans = 0.0, 0.0, 0.0
    correct_bin, correct_trans, total = 0, 0, 0

    for images, labels_bin, labels_trans in loader:
        images = images.to(device)
        labels_bin = labels_bin.to(device)
        labels_trans = labels_trans.to(device)

        optimizer.zero_grad()
        out_bin, out_trans = model(images)

        loss_bin = criterion_bin(out_bin, labels_bin)
        loss_trans = criterion_trans(out_trans, labels_trans)
        total_loss = w1 * loss_bin + w2 * loss_trans

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += total_loss.item() * batch_size
        running_loss_bin += loss_bin.item() * batch_size
        running_loss_trans += loss_trans.item() * batch_size

        correct_bin += (torch.max(out_bin, 1)[1] == labels_bin).sum().item()
        correct_trans += (torch.max(out_trans, 1)[1] == labels_trans).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "loss_bin": running_loss_bin / total,
        "loss_trans": running_loss_trans / total,
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
        images = images.to(device)
        labels_bin = labels_bin.to(device)
        labels_trans = labels_trans.to(device)

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
def train_singletask_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=1.0):
    """Train one epoch for a single-task model."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        correct += (torch.max(out, 1)[1] == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_singletask(model, loader, criterion, device):
    """Evaluate a single-task model."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        running_loss += criterion(out, labels).item() * images.size(0)
        correct += (torch.max(out, 1)[1] == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


# =============================================================================
# Full Multi-Task Training Pipeline
# =============================================================================
def train_multitask_model(model, train_loader, val_loader, config, w1=0.5, w2=0.5, tag="multitask"):
    """Full training loop with LR scheduler, early stopping, and checkpointing."""
    device = config["device"]
    criterion_bin = nn.CrossEntropyLoss()
    criterion_trans = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_score = 0.0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Training [{tag}] | w1={w1}, w2={w2} | {config['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(config["epochs"]):
        # Train
        train_metrics = train_multitask_epoch(
            model, train_loader, optimizer, criterion_bin, criterion_trans,
            device, w1, w2, config["grad_clip_norm"]
        )

        # Validate
        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, device)

        # Step scheduler
        scheduler.step()

        # Combined val score for checkpointing
        val_score = (val_metrics["acc_bin"] + val_metrics["acc_trans"]) / 2

        # Log
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

        # Print progress
        print(f"\nEpoch [{epoch+1}/{config['epochs']}] "
              f"Loss: {train_metrics['loss']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  [Train] Real/Fake: {train_metrics['acc_bin']*100:.2f}% | "
              f"Transform: {train_metrics['acc_trans']*100:.2f}%")
        print(f"  [Val]   Real/Fake: {val_metrics['acc_bin']*100:.2f}% | "
              f"Transform: {val_metrics['acc_trans']*100:.2f}%")

        # Best model checkpoint
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = deepcopy(model.state_dict())
            ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{tag}.pth")
            torch.save(best_model_state, ckpt_path)
            print(f"  ★ New best model saved ({val_score*100:.2f}%)")

        # Early stopping
        early_stop(val_score)
        if early_stop.should_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model (val score: {best_val_score*100:.2f}%)")

    return model, logger


# =============================================================================
# Full Single-Task Training Pipeline
# =============================================================================
def train_singletask_model(model, train_loader, val_loader, config, task_name="binary"):
    """Full training loop for a single-task baseline."""
    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_acc = 0.0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Training Unimodal Baseline [{task_name}] | {config['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_singletask_epoch(
            model, train_loader, optimizer, criterion, device, config["grad_clip_norm"]
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
