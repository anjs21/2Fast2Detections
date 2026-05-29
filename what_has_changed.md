# What Has Changed

This document describes every modification made to `code/` to introduce differential learning rates, backbone freezing, and dynamic loss balancing (Uncertainty Weighting and GradNorm).

---

## File-by-file Changes

### `code/config.py`

Six new keys added to `CONFIG`:

| Key | Default | Description |
|-----|---------|-------------|
| `freeze_epochs` | `3` | Epochs to keep the backbone frozen at startup. `0` = no freeze. |
| `backbone_lr` | `1e-5` | Learning rate for backbone parameters after unfreeze. |
| `head_lr` | `1e-4` | Learning rate for both task heads (and `log_var` when using uncertainty). |
| `loss_weighting_method` | `"static"` | Which loss balancing method to use: `"static"`, `"uncertainty"`, or `"gradnorm"`. |
| `gradnorm_alpha` | `1.5` | GradNorm task-asymmetry exponent. Higher values steer training more aggressively toward the harder task. `0` equalises gradient norms regardless of relative loss. The original paper's `0.12` is too conservative for two classification tasks. |
| `gradnorm_weight_lr` | `1e-3` | Learning rate for the GradNorm task-weight optimizer (separate Adam, internal to `GradNormBalancer`). |

**Constraint:** `"gradnorm"` is incompatible with `freeze_epochs > 0`. A runtime assertion in `train_multitask_model` enforces this before any training begins.

---

### `code/models.py`

`get_backbone()` now returns a third value — a reference to the **shared representation layer** that GradNorm uses to compute per-task gradient norms. This is resolved once at construction time so no architecture-specific logic is needed at training time:

- `resnet18` / `resnet50`: `base.layer4[-1]`
- `efficientnet_b0`: `base.features[-1]`

`MultiTaskModel` stores this as `self.shared_layer` so `GradNormBalancer.forward` can call `model.shared_layer` directly.

`SingleTaskModel` discards the third return value with `_, = get_backbone(...)` to avoid a `ValueError: too many values to unpack`.

---

### `code/training.py`

#### New classes: `StaticWeighter`, `UncertaintyLoss`, `GradNormBalancer`

All three implement the same interface: `weighter(L1, L2, model)` returns the combined scalar loss. This keeps `train_multitask_epoch` identical across methods.

**`StaticWeighter`** — plain class, no parameters:
```python
total_loss = w1 * L1 + w2 * L2
```

**`UncertaintyLoss`** (`nn.Module`) — Kendall et al. homoscedastic uncertainty weighting. Owns two learnable log-variance scalars (`self.log_var`):
```
total_loss = 0.5 * exp(-log_var[0]) * L1 + 0.5 * log_var[0]
           + 0.5 * exp(-log_var[1]) * L2 + 0.5 * log_var[1]
```
The `exp(-log_var)` term acts as a precision weight; the `log_var` term is the regulariser that prevents the network from setting all precision to zero. Both scalars are included in the main AdamW optimizer as a fourth param group (at `head_lr`) so they are updated alongside the heads.

**`GradNormBalancer`** (`nn.Module`) — Chen et al. GradNorm. Owns two task weights (`self.weights`) and a separate `Adam` optimizer for them (`self.weight_optimizer`). The full update per batch:
1. Compute per-task gradient norms `G1`, `G2` on `model.shared_layer` using `autograd.grad(..., create_graph=True, retain_graph=True)`.
2. Compute targets from the running loss ratio raised to the power `alpha`.
3. Backward through the GradNorm loss to update `self.weights` via `weight_optimizer`.
4. Clamp weights to `≥ 0`, then renormalise so they sum to `2.0`.
5. Return `w1.detach() * L1 + w2.detach() * L2` — detached weights prevent the weight graph from contaminating model gradients.

`GradNormBalancer.to(device)` overrides `nn.Module.to()` to reinitialise `weight_optimizer` after the device move, keeping the optimizer state consistent.

All three classes expose `get_weights() -> (float, float)` for logging:
- `StaticWeighter`: `(w1, w2)`
- `UncertaintyLoss`: `(log_var[0], log_var[1])`
- `GradNormBalancer`: `(weights[0], weights[1])`

#### Updated: `train_multitask_epoch`

Signature change: `w1, w2` replaced by a single `loss_weighter` callable.

**Critical ordering change:** `optimizer.zero_grad()` now comes **after** `loss_weighter(...)`, not before. GradNorm's internal `gradnorm_loss.backward()` leaks gradients into model parameters (via `create_graph=True`). Calling `zero_grad()` after the weighter clears both previous-batch gradients and GradNorm leakage before the main `total_loss.backward()`.

For `StaticWeighter` and `UncertaintyLoss`, `loss_weighter()` does no backward pass, so the post-call `zero_grad()` simply clears the previous batch — identical in effect to the original code.

#### Updated: `train_multitask_model`

Signature change: `w1, w2` replaced by `loss_weighter=None` (defaults to `StaticWeighter(0.5, 0.5)`).

**Backbone freezing — no-rebuild design:**

All three param groups (backbone, binary head, transform head) are registered in `AdamW` from epoch 0. During freeze, the backbone group is dormant (`requires_grad=False`). At epoch `freeze_epochs`, only `requires_grad` is flipped — the optimizer is never rebuilt. This preserves AdamW moment buffers for the head parameters across the freeze/unfreeze boundary, avoiding the loss spike that a rebuild would cause.

**`LambdaLR` cosine schedule:**

`CosineAnnealingLR` is replaced by `LambdaLR` with per-group lambda functions:

```python
def backbone_lambda(epoch):
    if epoch < freeze:
        return 0.0          # backbone lr = 0 during freeze
    progress = (epoch - freeze) / max(total - freeze, 1)
    return 0.5 * (1 + cos(π * progress))   # cosine from unfreeze onward

def head_lambda(epoch):
    return 0.5 * (1 + cos(π * epoch / total))   # full cosine from epoch 0
```

This eliminates manual lr mutation and scheduler reconstruction.

**`UncertaintyLoss` param group construction:**

`UncertaintyLoss` is created before the optimizer so its `log_var` parameters can be included in a fourth param group. If the param group count differed between methods, `LambdaLR` would raise `ValueError: len(lr_lambda) != len(optimizer.param_groups)`. The fix: always build 3 model groups, then conditionally append the 4th group + lambda only when uncertainty is active.

**Per-epoch logging:**

`weight_1` and `weight_2` are logged to `TrainingLogger` each epoch via `loss_weighter.get_weights()`. The logged LR is from `param_groups[1]` (head lr) rather than `param_groups[0]` (backbone lr, which is `0` during freeze).

---

### `code/stage_ablation.py`

`weight_configs` extended from 5 static entries to 7 entries that include the two dynamic methods:

```python
weight_configs = [
    ("static",      1.0,  0.0,  "binary_only"),
    ("static",      0.75, 0.25, "binary_dominant"),
    ("static",      0.5,  0.5,  "equal"),
    ("static",      0.25, 0.75, "transform_dominant"),
    ("static",      0.0,  1.0,  "transform_only"),
    ("uncertainty", None, None, "uncertainty"),
    ("gradnorm",    None, None, "gradnorm"),
]
```

The loop now branches on `method` to construct the appropriate `loss_weighter` and calls `train_multitask_model(..., loss_weighter=loss_weighter, ...)`. The Pareto plot annotations and bar-chart x-labels use the `label` string instead of raw `w1`/`w2` floats, so dynamic methods appear as labelled points on the same frontier.

---

### `code/stage_multimodal.py`

`run_multimodal_training` gains an optional `loss_weighter=None` parameter that is passed through to `train_multitask_model`. The training curve title reflects the active method name.

---

### `code/main.py`

Constructs the `loss_weighter` from `CONFIG["loss_weighting_method"]` before calling `run_multimodal_training`, and passes it in:

```python
method = CONFIG["loss_weighting_method"]
if method == "uncertainty":
    loss_weighter = UncertaintyLoss()
elif method == "gradnorm":
    loss_weighter = GradNormBalancer(CONFIG["gradnorm_alpha"], CONFIG["gradnorm_weight_lr"])
else:
    loss_weighter = StaticWeighter(0.5, 0.5)
```

---

## How to Configure and Run

### Static weighting (default)

```python
CONFIG["loss_weighting_method"] = "static"
CONFIG["freeze_epochs"] = 3
```

The standard case. Set `freeze_epochs=0` to disable backbone freezing.

### Uncertainty weighting

```python
CONFIG["loss_weighting_method"] = "uncertainty"
CONFIG["freeze_epochs"] = 3   # fine to freeze with uncertainty
```

`log_var` starts at `0` (equal precision for both tasks) and diverges as training reveals task difficulty asymmetry.

### GradNorm

```python
CONFIG["loss_weighting_method"] = "gradnorm"
CONFIG["freeze_epochs"] = 0   # REQUIRED — non-zero freeze raises AssertionError
CONFIG["gradnorm_alpha"] = 1.5
CONFIG["gradnorm_weight_lr"] = 1e-3
```

`gradnorm_alpha` controls how aggressively training is steered toward the harder task:
- `alpha = 0`: equalises gradient norms regardless of relative loss (no task-difficulty awareness)
- `alpha = 1.5` (default): moderately favours the task with higher relative loss
- Higher values (2–3): very aggressive rebalancing, may destabilise early training

Task weights are renormalised to sum `2.0` after each batch, keeping scale parity with static baselines where `w1 + w2 = 1.0` (note: GradNorm sums to 2 by convention since each weight starts at 1).

---

## Mathematical Background

### Uncertainty Weighting (Kendall et al., 2018)

For tasks with losses `L_1`, `L_2` and homoscedastic noise parameters `σ_1`, `σ_2`:

```
L = (1/2σ_1²) * L_1 + log σ_1 + (1/2σ_2²) * L_2 + log σ_2
```

Parameterising `s_i = log σ_i²` (i.e. `log_var[i]`) for numerical stability:

```
L = 0.5 * exp(-s_1) * L_1 + 0.5 * s_1
  + 0.5 * exp(-s_2) * L_2 + 0.5 * s_2
```

The `exp(-s_i)` factor down-weights high-uncertainty tasks; the `0.5 * s_i` regulariser prevents degenerate collapse to infinite uncertainty.

### GradNorm (Chen et al., 2018)

Let `w_i` be learnable task weights, `G_i = ||∇_W (w_i L_i)||` the gradient norm on the shared layer `W`, and `r_i = L_i / L_i^(0)` the relative loss (loss at epoch vs. initial loss).

The GradNorm target for task `i`:
```
G̅ = mean(G_i)
r̅ = mean(r_i)
G_i^target = G̅ * (r_i / r̅)^alpha
```

The auxiliary loss that updates task weights:
```
L_grad = Σ_i |G_i - G_i^target|
```

`L_grad` is minimised w.r.t. `w_i` only (model parameters are treated as fixed during this step). After each update, weights are renormalised: `w_i ← w_i * 2 / Σ w_j`.
