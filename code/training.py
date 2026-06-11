"""
Training utilities: EarlyStopping, TrainingLogger, train/eval epoch functions,
and full training pipelines for both multi-task and single-task models.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
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
# Loss Weighters
# =============================================================================
class StaticWeighter:
    """Fixed scalar weighting of the two task losses."""

    def __init__(self, w1=0.5, w2=0.5):
        self.w1 = w1
        self.w2 = w2

    def __call__(self, L1, L2, model=None):
        return self.w1 * L1 + self.w2 * L2

    def get_weights(self):
        return self.w1, self.w2


class UncertaintyLoss(nn.Module):
    """Kendall et al. homoscedastic uncertainty weighting."""

    def __init__(self):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(2))

    def forward(self, L1, L2, model=None):
        precision1 = torch.exp(-self.log_var[0])
        precision2 = torch.exp(-self.log_var[1])
        return (0.5 * precision1 * L1 + 0.5 * self.log_var[0] +
                0.5 * precision2 * L2 + 0.5 * self.log_var[1])

    def get_weights(self):
        return self.log_var[0].item(), self.log_var[1].item()


class GradNormBalancer(nn.Module):
    """Chen et al. GradNorm dynamic task weighting."""

    def __init__(self, alpha, lr):
        super().__init__()
        self.alpha = alpha
        self.lr = lr
        self.weights = nn.Parameter(torch.ones(2))
        self.weight_optimizer = optim.Adam([self.weights], lr=lr)
        self.L0 = None

    def to(self, *args, **kwargs):
        # Reinitialize weight_optimizer after device move so its state stays consistent.
        super().to(*args, **kwargs)
        self.weight_optimizer = optim.Adam([self.weights], lr=self.lr)
        return self

    def forward(self, L1, L2, model):
        W = list(model.shared_layer.parameters())

        # Per-task gradient norms computed without create_graph (retain_graph keeps L1/L2 graph for main backward)
        g1 = autograd.grad(L1, W, retain_graph=True)
        g2 = autograd.grad(L2, W, retain_graph=True)
        G1_norm = torch.norm(torch.stack([g.norm() for g in g1]))
        G2_norm = torch.norm(torch.stack([g.norm() for g in g2]))

        # Multiply by weight variables to create a simple graph containing only self.weights
        G1 = self.weights[0] * G1_norm
        G2 = self.weights[1] * G2_norm

        # GradNorm targets (detached — treated as constants)
        G_bar = ((G1 + G2) / 2).detach()
        if self.L0 is None:
            self.L0 = (L1.detach(), L2.detach())
        r1 = L1.detach() / self.L0[0]
        r2 = L2.detach() / self.L0[1]
        r_bar = (r1 + r2) / 2
        G_target_1 = (G_bar * (r1 / r_bar) ** self.alpha).detach()
        G_target_2 = (G_bar * (r2 / r_bar) ** self.alpha).detach()

        # Update task weights via GradNorm loss
        gradnorm_loss = (G1 - G_target_1).abs() + (G2 - G_target_2).abs()
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        self.weight_optimizer.step()

        # Clamp before renorm: negative weights would flip signs after division
        with torch.no_grad():
            self.weights.clamp_(min=0.0)
            self.weights.data *= 2.0 / self.weights.sum().clamp(min=1e-8)

        # Detached weights: model gradients must not flow through the weight graph
        return self.weights[0].detach() * L1 + self.weights[1].detach() * L2

    def get_weights(self):
        return self.weights[0].item(), self.weights[1].item()


# =============================================================================
# Multi-Task Epoch Functions
# =============================================================================
def train_multitask_epoch(model, loader, optimizer, criterion_bin, criterion_trans,
                          device, loss_weighter, grad_clip_norm=1.0):
    """Train one epoch for the multi-task model."""
    model.train()
    running_loss, running_loss_bin, running_loss_trans = 0.0, 0.0, 0.0
    correct_bin, correct_trans, total = 0, 0, 0

    for images, labels_bin, labels_trans in loader:
        images = images.to(device)
        labels_bin = labels_bin.to(device)
        labels_trans = labels_trans.to(device)

        out_bin, out_trans = model(images)

        loss_bin = criterion_bin(out_bin, labels_bin)
        loss_trans = criterion_trans(out_trans, labels_trans)

        # GradNorm runs its internal backward here and may leak into model .grad
        total_loss = loss_weighter(loss_bin, loss_trans, model)

        # zero_grad AFTER loss_weighter: clears both previous-batch grads and any GradNorm leakage
        optimizer.zero_grad()
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
def train_multitask_model(model, train_loader, val_loader, config, loss_weighter=None, tag="multitask"):
    """Full training loop with differential LR, backbone freezing, LambdaLR, early stopping, and checkpointing."""
    device = config["device"]

    if loss_weighter is None:
        loss_weighter = StaticWeighter(0.5, 0.5)

    # Move nn.Module weighers to device (GradNormBalancer.to() also reinits its weight_optimizer)
    if isinstance(loss_weighter, nn.Module):
        loss_weighter = loss_weighter.to(device)

    assert not (isinstance(loss_weighter, GradNormBalancer) and config.get("freeze_epochs", 0) > 0), (
        "GradNorm requires freeze_epochs=0: autograd.grad errors when backbone params have requires_grad=False."
    )

    criterion_bin = nn.CrossEntropyLoss()
    criterion_trans = nn.CrossEntropyLoss()

    freeze = config.get("freeze_epochs", 0)
    total_epochs = config["epochs"]
    backbone_lr = config.get("backbone_lr", config["lr"])
    head_lr = config.get("head_lr", config["lr"])

    # Freeze backbone for the initial epochs; all param groups are registered from epoch 0
    # so head AdamW moment buffers are never reset at unfreeze time.
    for p in model.backbone.parameters():
        p.requires_grad_(freeze == 0)

    def backbone_lambda(epoch):
        if epoch < freeze:
            return 0.0
        progress = (epoch - freeze) / max(total_epochs - freeze, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def head_lambda(epoch):
        return 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))

    param_groups = [
        {"params": list(model.backbone.parameters()),       "lr": backbone_lr},
        {"params": list(model.binary_head.parameters()),    "lr": head_lr},
        {"params": list(model.transform_head.parameters()), "lr": head_lr},
    ]
    lambdas = [backbone_lambda, head_lambda, head_lambda]

    # UncertaintyLoss log_var scalars learn alongside the heads
    if isinstance(loss_weighter, UncertaintyLoss):
        param_groups.append({"params": list(loss_weighter.parameters()), "lr": head_lr})
        lambdas.append(head_lambda)

    optimizer = optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    early_stop = EarlyStopping(patience=config["patience"])
    logger = TrainingLogger()

    best_val_score = 0.0
    best_model_state = None

    method_name = type(loss_weighter).__name__
    print(f"\n{'='*60}")
    print(f"Training [{tag}] | method={method_name} | freeze={freeze} epochs | {total_epochs} epochs total")
    print(f"{'='*60}")

    for epoch in range(total_epochs):
        # Unfreeze backbone at the designated epoch; scheduler handles the lr ramp-up
        if epoch == freeze and freeze > 0:
            for p in model.backbone.parameters():
                p.requires_grad_(True)

        train_metrics = train_multitask_epoch(
            model, train_loader, optimizer, criterion_bin, criterion_trans,
            device, loss_weighter, config["grad_clip_norm"]
        )

        val_metrics = evaluate_multitask(model, val_loader, criterion_bin, criterion_trans, device)
        scheduler.step()

        val_score = (val_metrics["acc_bin"] + val_metrics["acc_trans"]) / 2
        w1, w2 = loss_weighter.get_weights()

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
            lr=optimizer.param_groups[1]["lr"],  # head lr
            weight_1=w1,
            weight_2=w2,
        )

        print(f"\nEpoch [{epoch+1}/{total_epochs}] "
              f"Loss: {train_metrics['loss']:.4f} | LR: {optimizer.param_groups[1]['lr']:.6f}")
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
