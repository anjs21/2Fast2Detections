"""
Optuna search over the two classification-head architectures (approach B:
feature-cached).

The (trained) backbone is run ONCE over train+val to cache the fused pre-head
features (see models.*.forward_features). Each Optuna trial then trains only the
two small MLP heads on those cached features and is scored on VALIDATION accuracy
-- the held-out TEST set is never touched. Hundreds of trials run in minutes.

Run on a GPU node (feature extraction is one forward pass over the data):

    python optuna_heads.py --trials 60 \
        --checkpoint ../checkpoints/final_multitask_model.pth

IMPORTANT: build this with the SAME CONFIG (backbone, dual_stream, ...) that
produced the checkpoint, so the cached features come from the trained backbone.
The script warns if backbone weights fail to load (it then falls back to an
ImageNet-init backbone, a weaker proxy).

Prints the best binary_head / transform_head dicts, ready to paste into config.py.
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna

from config import (
    CONFIG, CHECKPOINTS_DIR, ORIGINAL_TRAIN_DIR, ORIGINAL_VAL_DIR, TEST_SUBSET_DIR,
    BINARY_MAP, TRANSFORM_MAP, download_train_val_data, download_test_data,
)
from data import build_splits, MultiTaskDataset, get_val_transform
from models import MultiTaskModel, DualStreamMultiTaskModel, _make_head
from training import compute_class_weights

HIDDEN_CHOICES = [128, 256, 512, 1024]


def build_backbone_model():
    """Same architecture selection as the training stages (heads are irrelevant
    here -- only forward_features is used)."""
    if CONFIG.get("dual_stream"):
        return DualStreamMultiTaskModel(
            rgb_backbone_name=CONFIG["backbone"],
            noise_backbone_name=CONFIG.get("noise_backbone", "resnet18"),
            trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
        )
    return MultiTaskModel(
        backbone_name=CONFIG["backbone"],
        trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
    )


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, ys_bin, ys_trans = [], [], []
    for images, labels_bin, labels_trans in loader:
        f = model.forward_features(images.to(device, non_blocking=True))
        feats.append(f.float().cpu())
        ys_bin.append(labels_bin)
        ys_trans.append(labels_trans)
    return torch.cat(feats), torch.cat(ys_bin), torch.cat(ys_trans)


def suggest_head(trial, name):
    n_layers = trial.suggest_int(f"{name}_layers", 0, 3)
    dims = [trial.suggest_categorical(f"{name}_w{i}", HIDDEN_CHOICES) for i in range(n_layers)]
    dropout = trial.suggest_float(f"{name}_dropout", 0.0, 0.6)
    return {"hidden_dims": dims, "dropout": dropout}


def head_dict_from_params(params, name):
    n_layers = params[f"{name}_layers"]
    return {
        "hidden_dims": [params[f"{name}_w{i}"] for i in range(n_layers)],
        "dropout": round(params[f"{name}_dropout"], 3),
    }


def train_eval_heads(bin_cfg, trans_cfg, cache, weights, device, epochs, batch_size=256):
    """Train both MLP heads on cached features; return (val_bin_acc, val_trans_acc)."""
    fdim = cache["train_f"].shape[1]
    bin_head = _make_head(fdim, 2, **bin_cfg).to(device)
    trans_head = _make_head(fdim, 3, **trans_cfg).to(device)

    opt = torch.optim.AdamW(list(bin_head.parameters()) + list(trans_head.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    crit_b = nn.CrossEntropyLoss(weight=weights["bin"].to(device))
    crit_t = nn.CrossEntropyLoss(weight=weights["trans"].to(device))

    Xtr = cache["train_f"].to(device)
    ybtr, yttr = cache["train_b"].to(device), cache["train_t"].to(device)
    Xva = cache["val_f"].to(device)
    ybva, ytva = cache["val_b"].to(device), cache["val_t"].to(device)

    n = Xtr.shape[0]
    for _ in range(epochs):
        bin_head.train(); trans_head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit_b(bin_head(Xtr[idx]), ybtr[idx]) + crit_t(trans_head(Xtr[idx]), yttr[idx])
            loss.backward()
            opt.step()

    bin_head.eval(); trans_head.eval()
    with torch.no_grad():
        acc_b = (bin_head(Xva).argmax(1) == ybva).float().mean().item()
        acc_t = (trans_head(Xva).argmax(1) == ytva).float().mean().item()
    return acc_b, acc_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40, help="head-training epochs per trial")
    ap.add_argument("--checkpoint", default=os.path.join(CHECKPOINTS_DIR, "final_multitask_model.pth"))
    args = ap.parse_args()
    device = CONFIG["device"]

    # ---- data (train + val only; test stays untouched) ----
    download_train_val_data()
    download_test_data()
    train_df, val_df, _ = build_splits(
        ORIGINAL_TRAIN_DIR, ORIGINAL_VAL_DIR, TEST_SUBSET_DIR,
        subset_per_class=CONFIG["subset_per_class"], seed=CONFIG["seed"],
        val_frac=CONFIG["val_frac"], test_frac=CONFIG["test_frac"], csv_path=None,
    )
    tfm = get_val_transform(CONFIG["img_size"], CONFIG["backbone"])  # no aug, deterministic
    mk = lambda df: DataLoader(MultiTaskDataset(df, tfm), batch_size=CONFIG["batch_size"],
                               shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # ---- trained backbone ----
    model = build_backbone_model().to(device)
    if os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device)
        res = model.load_state_dict(sd, strict=False)
        bb_missing = [k for k in res.missing_keys
                      if any(t in k for t in ("backbone", "bayar", "noise"))]
        print(f"[ckpt] {args.checkpoint}: missing={len(res.missing_keys)} "
              f"unexpected={len(res.unexpected_keys)}")
        if bb_missing:
            print(f"[WARN] {len(bb_missing)} backbone/stream tensors NOT loaded -- the config "
                  f"likely does not match the checkpoint. Features fall back to ImageNet init "
                  f"(weaker proxy). Match CONFIG to the run that produced the checkpoint.")
    else:
        print(f"[ckpt] {args.checkpoint} not found -> ImageNet-init backbone (weaker proxy).")

    # ---- cache features once ----
    print("Extracting features (one forward pass over train + val)...")
    train_f, train_b, train_t = extract_features(model, mk(train_df), device)
    val_f, val_b, val_t = extract_features(model, mk(val_df), device)
    print(f"  cached train {tuple(train_f.shape)} | val {tuple(val_f.shape)}")
    cache = {"train_f": train_f, "train_b": train_b, "train_t": train_t,
             "val_f": val_f, "val_b": val_b, "val_t": val_t}
    weights = {"bin": compute_class_weights(train_df, "binary_label", BINARY_MAP, "cpu"),
               "trans": compute_class_weights(train_df, "transform_label", TRANSFORM_MAP, "cpu")}

    # ---- baseline: the current [256] head, for reference ----
    base = {"hidden_dims": [256], "dropout": 0.3}
    b_acc, t_acc = train_eval_heads(base, base, cache, weights, device, args.epochs)
    print(f"\n[baseline [256]/[256]] val binary {b_acc*100:.2f}% | transform {t_acc*100:.2f}% "
          f"| score {(b_acc+t_acc)/2*100:.2f}%")

    # ---- Optuna study (maximize mean val accuracy of the two heads) ----
    def objective(trial):
        bin_cfg = suggest_head(trial, "bin")
        trans_cfg = suggest_head(trial, "trans")
        acc_b, acc_t = train_eval_heads(bin_cfg, trans_cfg, cache, weights, device, args.epochs)
        trial.set_user_attr("bin_acc", acc_b)
        trial.set_user_attr("trans_acc", acc_t)
        return (acc_b + acc_t) / 2

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=CONFIG["seed"]))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    bt = study.best_trial
    print("\n" + "=" * 60)
    print("BEST HEAD CONFIGURATION (val)")
    print("=" * 60)
    print(f"val score {bt.value*100:.2f}% | binary {bt.user_attrs['bin_acc']*100:.2f}% | "
          f"transform {bt.user_attrs['trans_acc']*100:.2f}%")
    print("\nPaste into config.py:")
    print(f'    "binary_head": {head_dict_from_params(bt.params, "bin")},')
    print(f'    "transform_head": {head_dict_from_params(bt.params, "trans")},')


if __name__ == "__main__":
    main()
