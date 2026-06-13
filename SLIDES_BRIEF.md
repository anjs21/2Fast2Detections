# Slide-Generation Brief — "Proposed Method" & "Experimental Setup & Evaluation"

> **HOW TO USE:** Paste this whole file into Claude. If you have the figures in
> `results/`, upload them too and Claude will place them; otherwise Claude should
> generate clean diagrams/charts itself.

---

## INSTRUCTIONS FOR CLAUDE (read first)

You are producing a **professional 16:9 presentation deck** covering two sections of a
computer-vision project: **(A) Proposed Method** and **(B) Experimental Setup &
Evaluation**. Deliverables and rules:

- **Format:** Render as a self-contained **reveal.js HTML artifact** (one file). If the
  user prefers PowerPoint, instead output `python-pptx` code that builds the same deck.
- **Audience:** technical (graduate ML course / reviewers). Assume they know CNNs/transformers.
- **Style:** one idea per slide, ≤ 6 bullets per slide, short phrases not sentences,
  prefer diagrams/tables over text. Clean, modern, high-contrast. Consistent accent color.
  Title + concise body + optional figure. Add the provided **speaker notes** to each slide.
- **Do not invent numbers.** Use the content below verbatim (you may tighten wording).
  Where a value is marked `[FILL]`, leave a clearly-styled placeholder.
- **Length:** ~12 content slides + 2 section-divider slides. Number the slides.
- Generate simple **diagrams** (architecture, dual-stream, data-split) as inline SVG/HTML
  boxes-and-arrows where no figure file is provided.

---

## PROJECT CONTEXT (for your understanding — don't dump all of this onto slides)

**Title:** *Joint Detection of AI-Generated Images and Post-Processing Alterations in
Real-World Scenarios.*

**Problem.** Given an image, jointly predict **(1) real vs AI-generated** (binary) and
**(2) the post-processing transform** it underwent — `original`, `transmitted`
(shared over the internet/social media → recompressed), or `redigitalized`
(screenshotted / re-photographed). Real-world AI images are usually *degraded* by these
transforms, which weakens detection — hence studying both jointly.

**Data.** RRDataset (real + AI images; each source scene exists in multiple transform
versions). Balanced subset ≈ **1000 images/class**; splits ≈ **4,800 train / 1,040 val /
1,040 test**, images 224×224.

**Final proposed method.** A **multi-task** network: a shared **ConvNeXt-Base** backbone
(fine-tuned from ImageNet) feeding **two MLP heads** (binary 2-class, transform 3-class),
trained with a class-weighted joint loss `L = w1·CE_binary + w2·CE_transform`. Two design
ideas: **(i)** a **dual-stream forensic branch** — a Bayar constrained high-pass layer →
ResNet-18 on the noise residual, fused with the RGB features, to capture high-frequency
generation fingerprints; **(ii)** **forensics-aware preprocessing** — native-pixel crops
(never downscale) + 5-crop test-time augmentation, to preserve the fragile artifacts.
Evaluation uses a **group-aware, leak-free** train/val/test split.

**Headline results (held-out test, with TTA):** **~92–93% real/fake accuracy, ~98%
transform accuracy, ROC-AUC ≈ 0.98.** Hardest case: **AI images that were
re-digitized** (their high-frequency fingerprint is physically destroyed).

---

## DESIGN GUIDANCE
- Accent color: a single deep blue or teal; neutral grays for body.
- Sans-serif font (e.g., Inter / Helvetica). Generous whitespace.
- Tables: header row shaded, monospaced numbers, bold the best row.
- Each results slide: lead with the number, support with the figure.

---

# SECTION A — PROPOSED METHOD

### Divider slide — "Proposed Method"

---

### Slide 1 — Problem Formulation
- **One image → two predictions:** Real vs AI-generated **and** transform type
  (original / transmitted / redigitalized).
- Real-world AI images arrive **degraded** (shared online, screenshotted) → detection must survive post-processing.
- **Proposal:** a single **multi-task** model that learns both jointly from shared features.
- *Visual:* diagram — `image → shared backbone → {Head A: Real/Fake} + {Head B: transform}`.
- *Speaker notes:* The two tasks are coupled — the transform is exactly what makes detection hard, so we model them together rather than separately.

### Slide 2 — Architecture: Shared Backbone + Two Heads
- Shared **ConvNeXt-Base** backbone (full fine-tune from ImageNet).
- Two lightweight **MLP heads**: binary (2-class) and transformation (3-class).
- **Joint loss:** `L = w1·CE_binary + w2·CE_transform`, class-weighted (handles mild imbalance).
- Shared representation = the multi-task "unified model".
- *Visual:* architecture block diagram (backbone → split into two heads).
- *Speaker notes:* Heads are deliberately small; the shared trunk does the heavy lifting. Loss weights w1/w2 trade off the two tasks (ablated later).

### Slide 3 — Design Idea 1: Forensic Noise-Residual Stream (Dual-Stream)
- **Insight:** the real-vs-AI signal lives in **high-frequency generation artifacts**, not image content.
- Add a **Bayar constrained high-pass** layer (learns a residual filter, kernel sums to 0) → **ResNet-18** on the residual.
- **Fuse** RGB features ⊕ noise features → both heads.
- Targets the hardest cases (degraded images) where RGB content cues weaken.
- *Visual:* two parallel streams (RGB backbone; Bayar→ResNet-18 on residual) → concat → heads; small residual-image thumbnail.
- *Speaker notes:* Standard forensics technique (Bayar & Stamm; RGB-N). We quantify its contribution in the ablations — it mainly helps the `transmitted` class.

### Slide 4 — Design Idea 2: Forensics-Aware Preprocessing
- Standard resize/augmentation **resamples away** the very artifacts the task depends on.
- Our choices: **native-pixel crops** (crop, don't downscale) + degradation-style augmentation.
- **Test-time augmentation:** average predictions over **5 deterministic crops**.
- This was the decisive fix for transform classification (it had collapsed under naïve resizing).
- *Visual:* before/after schematic — "downscale whole image (artifacts lost)" vs "native-resolution crop (artifacts preserved)".
- *Speaker notes:* The transform label *is* a low-level artifact; downscaling at eval time destroyed it. Matching train/test scale at native resolution recovered it.

### Slide 5 — Training Recipe (at a glance)
- Backbone: ConvNeXt-Base, 224×224, 10 epochs, batch 32, AdamW, lr 1e-4.
- Class-weighted cross-entropy; mixed precision (AMP); cosine LR; gradient clipping.
- **Early stopping & model selection on validation**; best checkpoint restored.
- ~12 min/run on a single A100.
- *Visual:* compact table of hyperparameters.
- *Speaker notes:* Lightweight and reproducible; everything below uses this recipe.

---

# SECTION B — EXPERIMENTAL SETUP & EVALUATION

### Divider slide — "Experimental Setup & Evaluation"

---

### Slide 6 — Dataset
- RRDataset: real + AI images, each source scene in **3 transform versions**.
- Classes: real / AI × {original, transmitted, redigitalized}; balanced ≈ 1000/class.
- Splits ≈ **4,800 train / 1,040 val / 1,040 test**.
- *Visual:* 2×3 grid of example images (real & fake × the 3 transforms) — use `results/visual_inference.png` if available.
- *Speaker notes:* `transmitted` = internet/social recompression; `redigitalized` = screenshot/re-photograph.

### Slide 7 — Leak-Free, Group-Aware Splitting (methodology highlight)
- **Risk:** the same source scene appears in multiple transforms → naïve splitting **leaks content** across train/test.
- **Fix:** split by **source-scene identity** (group-aware) → train/val/test fully disjoint, verified zero overlap.
- **Protocol:** **val = model selection / early stopping; test = final held-out report.**
- *Visual:* diagram — a scene's transform variants all kept on one side of the split.
- *Speaker notes:* This is what makes every later number trustworthy; we explicitly removed a ~6% content leak this introduced.

### Slide 8 — Evaluation Protocol & Metrics
- **Metrics:** Accuracy, Precision/Recall/F1; **ROC-AUC & Average Precision** for binary; confusion matrices.
- **Analyses:** per-transformation breakdown; cross-class (2×3) traces; unimodal-vs-multi-task; loss-weight ablation; multi-seed robustness.
- 5-crop **TTA** at test time; numbers reported on the held-out test set.
- *Visual:* small icon list or a one-row "metrics covered" strip.
- *Speaker notes:* AUC/AP are the standard AI-image-detection metrics; per-transform & cross-class reveal *where* failures concentrate.

### Slide 9 — Main Results
- **Real/Fake: ~92–93%  |  Transform: ~98%  |  ROC-AUC ≈ 0.98** (held-out test, TTA).
- Joint training does **not** hurt — slightly **helps** binary vs the unimodal baseline.
- Comparison table:

  | Model | Real/Fake Acc | Transform Acc |
  |---|---|---|
  | Unimodal (binary only) | ~92% | — |
  | Unimodal (transform only) | — | ~98% |
  | **Multi-task (joint)** | **~92–93%** | **~98%** |

- *Visual:* `results/unimodal_vs_multitask.png` + the two confusion matrices (`cm_Multi-Task_Real_Fake.png`, `cm_Multi-Task_Transformation.png`).
- *Speaker notes:* Best single run: 93.2% binary / 98.3% transform / AUC 0.980 (ConvNeXt-Base + dual-stream + TTA).

### Slide 10 — Where It Works and Where It Fails
- Per-transformation real/fake accuracy (best run): **Original 96.0% · Transmitted 91.1% · Redigitalized 91.0%**.
- Real-image accuracy stays high (95–99%); errors concentrate on **fake** images.
- **Hardest cell: fake + redigitalized (~83–86%)** — re-digitization physically destroys the high-frequency fingerprint (a *data* limit, not a model gap).
- *Visual:* `results/per_transformation_breakdown.png` + `results/cross_class_traces.png`.
- *Speaker notes:* This is the honest open problem; no backbone or stream we tried moved this cell.

### Slide 11 — Ablations & What We Learned
- **Loss weighting (val):** balanced weights give ~90% / ~98% together; extremes collapse one task (binary-only → transform 34%; transform-only → binary 48%). → tasks **don't strongly compete**.
- **Backbones:** ResNet50, ConvNeXt-Tiny/Base, CLIP ViT-L/14 all plateau at **89–93%** binary.
- **Head search:** 60-trial Optuna over head architectures → **0.8% total spread**; default head already optimal.
- **Seed robustness:** mean ± std over 3 seeds — Binary `[FILL]±[FILL]`, Transform `[FILL]±[FILL]`.
- **Takeaway:** the ceiling is the **data/signal**, not model capacity.
- *Visual:* `results/ablation_pareto.png` (loss-weight trade-off).
- *Speaker notes:* Convergence across very different backbones is the key evidence for a data ceiling.

### Slide 12 — Key Takeaways
- A single **multi-task** model jointly detects AI images **and** their post-processing at ~92% / ~98%.
- **Decisive factors:** forensics-aware preprocessing (preserve high-freq artifacts) + **leak-free, group-aware evaluation**.
- **Open challenge:** AI images that were **re-digitized** — signal is physically erased.
- *Visual:* one-line summary + a compact "method ✓ / evaluation ✓ / open problem" strip.
- *Speaker notes:* End on the honest ceiling — it frames future work (frequency-domain / robustness to re-digitization).

---

## FIGURE → SLIDE MAP (files live in `results/`)
- Slide 6: `visual_inference.png`
- Slide 9: `unimodal_vs_multitask.png`, `cm_Multi-Task_Real_Fake.png`, `cm_Multi-Task_Transformation.png`
- Slide 10: `per_transformation_breakdown.png`, `cross_class_traces.png` (optionally `confidence_distributions.png`)
- Slide 11: `ablation_pareto.png` (optionally `ablation_bars.png`, `curves_multitask.png`)

## PLACEHOLDERS TO FILL BEFORE PRESENTING
- Slide 11 seed robustness mean ± std (3-seed runs).
- Dataset exact counts on Slide 6 if you want precise numbers.
