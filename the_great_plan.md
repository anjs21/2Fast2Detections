# Multi-Task Learning Research Platform: Differential Learning & Dynamic Loss Balancing

This plan outlines modifications to the existing `code/` directory to make it a modular, research-oriented platform. The objective is to introduce flexible configuration, supporting early-epoch backbone freezing, differential learning rates, and dynamic loss balancing techniques (Uncertainty Weighting and GradNorm) to solve task competition.

## Proposed Changes

All changes are made directly to the files in `code/`. No files are copied or cloned.

---

### 1. Configuration & Main Orchestrator

#### [MODIFY] `code/config.py`
* Add researcher configuration parameters to `CONFIG`:
  * `"freeze_epochs"`: Number of initial epochs to freeze the backbone (e.g., `3`).
  * `"backbone_lr"`: Differential learning rate for the backbone parameters once unfrozen (e.g., `1e-5`).
  * `"head_lr"`: Learning rate for the task heads (e.g., `1e-4`).
  * `"loss_weighting_method"`: Method for scaling loss. Options: `"static"`, `"uncertainty"`, or `"gradnorm"`.
  * `"gradnorm_alpha"`: GradNorm task-asymmetry exponent. Default `1.5` (assumes equal task difficulty; `0` = equalizes gradient norms across tasks regardless of relative loss, higher = more aggressively steers towards the harder task). The original paper's value of `0.12` was tuned for a regression+classification pair and is too conservative for two classification tasks — it produces nearly static weighting and defeats the purpose.
  * `"gradnorm_weight_lr"`: Learning rate for the GradNorm weight optimizer (separate from the model optimizer), e.g., `1e-3`.
* **Constraint**: `"loss_weighting_method": "gradnorm"` is incompatible with `"freeze_epochs" > 0`. GradNorm calls `autograd.grad(..., model.shared_layer.parameters())`, which raises `RuntimeError` when those parameters have `requires_grad=False`. Add a runtime assertion at the start of training: `assert not (config["loss_weighting_method"] == "gradnorm" and config["freeze_epochs"] > 0)`. Use `"static"` or `"uncertainty"` during the freeze phase, or set `freeze_epochs=0` when using GradNorm.

#### [MODIFY] `code/main.py`
* Support switching between static baseline ablation sweeps and the new dynamic balancing methods via configuration.

---

### 2. Model & Training Engine

#### [MODIFY] `code/models.py`
* Extend `get_backbone()` to return a third value: a reference to the **shared representation layer** used by GradNorm. This is resolved once at construction time per architecture, avoiding any architecture-specific logic at training time:
  * `resnet18` / `resnet50`: `base.layer4[-1]`
  * `efficientnet_b0`: `base.features[-1]` (last MBConv block)
  ```python
  def get_backbone(name="resnet50"):
      if name == "resnet18":
          ...
          shared_layer = base.layer4[-1]
      elif name == "resnet50":
          ...
          shared_layer = base.layer4[-1]
      elif name == "efficientnet_b0":
          ...
          shared_layer = base.features[-1]
      return base, num_features, shared_layer
  ```
* Store the reference on `MultiTaskModel` as `self.shared_layer` so `GradNormBalancer` can call `model.shared_layer` directly without any backbone-specific branching.
* **`SingleTaskModel` must be updated to unpack the new third return value**, or it will raise `ValueError: too many values to unpack` at construction:
  ```python
  class SingleTaskModel(nn.Module):
      def __init__(self, ...):
          self.backbone, num_features, _ = get_backbone(backbone_name)  # discard shared_layer
  ```

#### [MODIFY] `code/training.py`

##### Loss Weighter Interface
Introduce a unified `loss_weighter` callable that replaces the old `w1, w2` float arguments in both `train_multitask_epoch` **and `train_multitask_model`**. Both function signatures must be updated — the outer pipeline function is what `stage_ablation.py` and `stage_multimodal.py` call directly. All three methods implement the same `__call__(L1, L2, model)` signature, keeping the training loop identical across methods:

```python
class StaticWeighter:
    def __init__(self, w1=0.5, w2=0.5): ...
    def __call__(self, L1, L2, model=None):
        return self.w1 * L1 + self.w2 * L2

class UncertaintyLoss(nn.Module):
    def __init__(self): ...          # owns log_var (trainable, 2 scalars)
    def forward(self, L1, L2, model=None):
        precision1 = torch.exp(-self.log_var[0])
        precision2 = torch.exp(-self.log_var[1])
        # log(sigma_i) = 0.5 * log(sigma_i^2) = 0.5 * log_var[i]
        return (0.5*precision1*L1 + 0.5*self.log_var[0] +
                0.5*precision2*L2 + 0.5*self.log_var[1])

class GradNormBalancer(nn.Module):
    def __init__(self, alpha, lr): ...  # owns weights (trainable) + weight_optimizer
    def forward(self, L1, L2, model): ...  # full GradNorm logic (see below)
```

The updated function signature:
```python
def train_multitask_epoch(model, loader, optimizer, loss_weighter, device, grad_clip_norm=1.0):
    ...
    total_loss = loss_weighter(loss_bin, loss_trans, model)
    ...
```

##### Backbone Freezing & Differential Learning
**Do not rebuild the optimizer at the unfreeze epoch.** Rebuilding resets AdamW moment buffers for the head parameters and causes a sharp loss spike as the optimizer re-warms. Instead, register all three param groups from epoch 0. During freeze, the backbone group is dormant (`requires_grad=False` + `lr=0` via the scheduler). At unfreeze, only flip `requires_grad` — the scheduler handles the lr:

```python
# Epoch 0: backbone starts frozen
for p in model.backbone.parameters():
    p.requires_grad_(False)

optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": config["backbone_lr"]},
    {"params": model.binary_head.parameters(),  "lr": config["head_lr"]},
    {"params": model.transform_head.parameters(), "lr": config["head_lr"]},
], weight_decay=config["weight_decay"])

# In training loop:
if epoch == config["freeze_epochs"]:
    for p in model.backbone.parameters():
        p.requires_grad_(True)
    # No optimizer rebuild — scheduler handles the lr ramp-up automatically
```

Replace `CosineAnnealingLR` with `LambdaLR` using per-group lambda functions. The backbone lambda returns `0.0` during the freeze phase then follows a cosine curve starting from `freeze_epochs`; the head lambda runs a full cosine over all epochs. This eliminates any need for manual lr mutation or scheduler reconstruction:

```python
import math

total = config["epochs"]
freeze = config["freeze_epochs"]

def backbone_lambda(epoch):
    if epoch < freeze:
        return 0.0
    progress = (epoch - freeze) / max(total - freeze, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def head_lambda(epoch):
    return 0.5 * (1.0 + math.cos(math.pi * epoch / total))

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=[backbone_lambda, head_lambda, head_lambda]
)
```

##### Uncertainty Weighting Module
Create `UncertaintyLoss` **before** building the optimizer and include its `log_var` parameters in a dedicated fourth param group. This ensures they are always part of the optimizer — if the module is created after the optimizer, its parameters are silently never updated.

**The optimizer and scheduler must always be built with the same number of param groups.** Merging heads + log\_var into one group would give 2 groups when uncertainty is active but 3 groups otherwise, causing `LambdaLR` to raise `ValueError: len(lr_lambda) != len(optimizer.param_groups)`. The fix is to build param groups and lambda lists together dynamically:

```python
# Create loss_weighter first so its params can enter the optimizer
if config["loss_weighting_method"] == "uncertainty":
    loss_weighter = UncertaintyLoss().to(device)
elif config["loss_weighting_method"] == "gradnorm":
    loss_weighter = GradNormBalancer(config["gradnorm_alpha"], config["gradnorm_weight_lr"])
else:
    loss_weighter = StaticWeighter(w1, w2)

# Build param groups — always 3 model groups, plus optional 4th for log_var
param_groups = [
    {"params": model.backbone.parameters(),       "lr": config["backbone_lr"]},
    {"params": model.binary_head.parameters(),    "lr": config["head_lr"]},
    {"params": model.transform_head.parameters(), "lr": config["head_lr"]},
]
lambdas = [backbone_lambda, head_lambda, head_lambda]

if config["loss_weighting_method"] == "uncertainty":
    param_groups.append({"params": list(loss_weighter.parameters()), "lr": config["head_lr"]})
    lambdas.append(head_lambda)

optimizer = optim.AdamW(param_groups, weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
```

The `log_var` scalars learn alongside the heads at `head_lr`, which is a reasonable default given their low dimensionality.

##### GradNorm Balancer
`GradNormBalancer` owns its own `weight_optimizer` (separate `Adam` instance) and performs the full GradNorm update internally on each `__call__`. This keeps the outer training loop identical to the other methods.

**GradNorm requires `freeze_epochs=0`.** It calls `autograd.grad(..., model.shared_layer.parameters())`, which errors when those params have `requires_grad=False`. The optimizer/scheduler construction block (see Uncertainty section) already instantiates `GradNormBalancer` before the optimizer — the startup assertion catches the invalid combination before any training begins.

The backward pass requires careful ordering to avoid graph conflicts. Use `torch.autograd.grad()` with `create_graph=True` and `retain_graph=True` for the GradNorm step, then call the main `backward()` separately. `create_graph=True` lets the GradNorm loss differentiate through the gradient norms into the task weights. `optimizer.zero_grad()` before the main backward clears any model-parameter gradients that leaked from the GradNorm pass:

```python
def forward(self, L1, L2, model):
    W = list(model.shared_layer.parameters())

    # Per-task gradient norms (retains graph for main backward)
    g1 = autograd.grad(self.weights[0] * L1, W, create_graph=True, retain_graph=True)
    g2 = autograd.grad(self.weights[1] * L2, W, create_graph=True, retain_graph=True)
    G1 = torch.norm(torch.stack([g.norm() for g in g1]))
    G2 = torch.norm(torch.stack([g.norm() for g in g2]))

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

    # Renormalize weights to sum=2.0 (maintains scale parity with static baselines)
    # Clamp before renorm: if a weight goes negative, dividing by a negative sum
    # flips the sign of both weights, producing invalid (negative) loss scaling.
    with torch.no_grad():
        self.weights.clamp_(min=0.0)
        self.weights.data *= 2.0 / self.weights.sum().clamp(min=1e-8)

    # Return weighted loss for main model backward
    # Use detached weights so model gradients are not contaminated by weight graph
    return self.weights[0].detach() * L1 + self.weights[1].detach() * L2
```

The caller (`train_multitask_epoch`) structures the inner loop as follows. **`optimizer.zero_grad()` must come AFTER `loss_weighter()`, not before.** GradNorm's `gradnorm_loss.backward()` (called inside `loss_weighter`) leaks gradients into model parameters via `create_graph=True`. Calling `zero_grad()` before `loss_weighter()` would leave those leaked grads in place, causing them to be silently mixed into the model update:

```python
for images, labels_bin, labels_trans in loader:
    images, labels_bin, labels_trans = images.to(device), labels_bin.to(device), labels_trans.to(device)

    # Forward pass
    out_bin, out_trans = model(images)
    loss_bin = criterion_bin(out_bin, labels_bin)
    loss_trans = criterion_trans(out_trans, labels_trans)

    # Loss weighting — GradNorm runs its internal backward here, may leak into model .grad
    total_loss = loss_weighter(loss_bin, loss_trans, model)

    # zero_grad AFTER loss_weighter: clears both previous-batch grads and GradNorm leakage
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
```

For `StaticWeighter` and `UncertaintyLoss`, `loss_weighter()` does no backward pass, so the post-`loss_weighter` `zero_grad()` simply clears the previous batch's grads — identical behavior to the original code.

---

### 3. Stages & Evaluation

#### [MODIFY] `code/stage_multimodal.py`
* Refactor to accept a `loss_weighter` instance (constructed in `main.py` based on `config["loss_weighting_method"]`) and pass it through to `train_multitask_model`.
* Log task weight trajectories over epochs for Uncertainty and GradNorm methods (log-variance values and `w1`/`w2` values respectively), so weight dynamics can be visualized.

#### [MODIFY] `code/stage_ablation.py`
* Extend `weight_configs` to include the two dynamic methods alongside the static weight sweep. This makes uncertainty and GradNorm appear as labeled points on the same Pareto frontier plot, enabling a direct comparison:
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
* Branch inside the loop on the method string vs. static floats to construct the appropriate `loss_weighter` and call `train_multitask_model` uniformly.

#### [NEW] `what_has_changed.md`
* Create a dedicated documentation file outlining:
  * Summary of every change made to each file in `code/`.
  * Details of the backbone freezing schedule, the no-rebuild optimizer design, and the `LambdaLR` cosine schedule.
  * Mathematical and architectural descriptions of both Uncertainty Weighting and GradNorm methods.
  * Guide on how to configure and run the new features, including how to tune `gradnorm_alpha`.

---

## Implementation Order

The following dependency chain should be respected:

1. `config.py` — add all new keys with corrected defaults (prerequisite for every other file that reads from `CONFIG`)
2. `models.py` — add `shared_layer` to `get_backbone()` and `MultiTaskModel` (prerequisite for GradNorm)
3. `training.py` — implement the three loss weighter classes and the unified `loss_weighter` interface
4. `training.py` — implement backbone freezing with the no-rebuild optimizer + `LambdaLR` scheduler
5. `training.py` — create `UncertaintyLoss` before optimizer construction (prerequisite for correct optimizer setup)
6. `training.py` — implement `GradNormBalancer.forward` with the double-backward ordering
7. `stage_ablation.py` — extend sweep to include dynamic methods
8. `stage_multimodal.py` — thread `loss_weighter` through and add weight logging
9. `what_has_changed.md` — write after all code is complete

---

## Verification Plan

### Automated Tests
Run sanity checks using a dry-run flag or small epoch counts (e.g., 1-2 epochs) to verify:

- **Freeze phase**: backbone parameters have `requires_grad=False`, their gradients are `None`, and their values do not change between epoch 0 and epoch `freeze_epochs - 1`.
- **Unfreeze transition**: at epoch `freeze_epochs`, backbone parameters receive non-zero gradients and update correctly at `backbone_lr`. Head AdamW moment buffers (`state[p]['exp_avg']`) are non-zero and continuous across the freeze/unfreeze boundary (confirming no optimizer rebuild occurred).
- **LR schedule**: backbone effective lr is `0.0` during freeze epochs and follows the cosine curve post-unfreeze. Head lr follows the full cosine from epoch 0.
- **Uncertainty**: `log_var` parameters appear in `optimizer.param_groups` from epoch 0, their values shift during training, and the effective per-task weights are not static.
- **GradNorm**: task weights `w1`, `w2` sum to `2.0` after each batch renormalization; gradient norms `G1`, `G2` tracked on `model.shared_layer` move toward their targets over training; no "graph freed" runtime errors occur.
- **EfficientNet backbone**: GradNorm runs without errors when `config["backbone"] = "efficientnet_b0"`, confirming the `shared_layer` resolution is architecture-agnostic.
- **Invalid config guard**: setting `loss_weighting_method="gradnorm"` with `freeze_epochs > 0` raises `AssertionError` immediately at startup, before any data is loaded.
- **Uncertainty formula**: after a few training steps, `log_var[0]` and `log_var[1]` shift away from their initial value of `0.0` and become non-equal, confirming the gradient through the `0.5 * log_var` regularizer term is non-zero and task-specific. (They need not move in opposite directions — if both task losses are similar in magnitude, both `log_var` values can move the same way; asymmetry in the tasks is what drives them apart.)
