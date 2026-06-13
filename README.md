# 2Fast2Detections: Joint Detection of AI-Generated Images & Post-Processing Alterations

This project implements a multi-task deep learning pipeline designed to jointly detect AI-generated images and identify the post-processing alterations (transformations) they have undergone (e.g., social media transmission, screen re-digitization). By predicting both tasks simultaneously using a shared representation, the model achieves better generalization and robustness against image degradation.

---

## Codebase Structure

The codebase is organized into six core Python modules and a batch submission script, keeping responsibilities decoupled and modular:

```text
├── globals.py         # Global hyperparameter configurations, paths, and seed setup
├── utils.py           # Supporting utilities (EarlyStopping, logger, downloaders, class weights)
├── data.py            # Dataset loaders, transforms/augmentation, and group-aware train/val/test splits
├── network.py         # Neural network architectures (Backbones, MLPs, Multi-Task, and Bayar Noise Stream)
├── train.py           # Training logic, training/evaluation loops, baseline runners, and main orchestrator
├── evaluation.py      # Metric reports, per-transformation breakdowns, heatmaps, and visual inference plotting
└── requirements.txt   # Python package dependencies
```

### Module Breakdown

*   **`globals.py`**: Holds path definitions, global hyperparameter configurations (`CONFIG` dictionary), label mapping directories, and seed initialization for strict reproducibility.
*   **`utils.py`**: Houses code support functions including model checkpoint loaders, download functions for train/val/test datasets, the `EarlyStopping` helper, the `TrainingLogger` tracker, and the class-weights utility for handling mild class imbalances.
*   **`data.py`**: Handles all aspects of data preparation. It defines custom PyTorch datasets, image processing transforms (including forensics-friendly crop-preserving augmentations), and a group-aware, stratified train/val/test split algorithm that prevents cross-transform scene leaks.
*   **`network.py`**: Defines the model architectures. It supports standard single-stream models (e.g. ResNet50, ConvNeXt, CLIP) as well as a dual-stream architecture fusing RGB features with a high-pass noise-residual stream (Bayar constrained conv layer).
*   **`train.py`**: Contains the training loop elements, baseline unimodal runners, and the main execution entry point (`__main__`). Running this script downloads the dataset, builds leak-free splits, trains baseline models, trains the joint multi-task model, and saves checkpoints.
*   **`evaluation.py`**: Provides tests and evaluation reports. Features per-transformation accuracy breakdown bar charts, cross-class trace analysis heatmaps, confidence distribution histograms, and visual prediction maps on sample test images.

---

## Getting Started

### Prerequisites

All package requirements are detailed in `requirements.txt`. The primary dependencies include:
*   `torch` & `torchvision`
*   `pandas`, `numpy`, and `pillow`
*   `scikit-learn`, `seaborn`, and `matplotlib`
*   `tqdm`

To install the dependencies manually:
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Locally / Interactively

To run the entire pipeline interactively, simply run:
```bash
python train.py
```
This script will perform the following steps sequentially:
1.  **Download & Extract Data**: Automatically fetches the original `train_val` dataset and the target categories of the `test_subset` if they are not already present on disk.
2.  **Generate Splits**: Computes source scene identities and splits the images into train, val, and test subsets to guarantee no cross-transform scene leakage.
3.  **Train Unimodal Baselines**: Fits independent single-task binary and transform classification baselines.
4.  **Train Multi-Task Model**: Trains the joint model using a weighted cross-entropy loss (incorporating inverse-frequency class weights).
5.  **Evaluate & Generate Plots**: Evaluates the model outputs on the held-out test split, saving confusion matrices, accuracy breakdowns, and training curves in `results/`.
6.  **Save Results**: Saves checkpoints in `checkpoints/` and the final metrics in `results/experiment_summary.json`.

## Key Configurations

Configurations are centralized in `globals.py` under the `CONFIG` dictionary. Notable options include:
*   `backbone`: The image feature extractor (e.g., `"convnext_base"`, `"resnet50"`, `"clip_vit_l14"`).
*   `dual_stream`: Fuses RGB features with a high-pass Bayar noise stream when set to `True`.
*   `trainable_backbone_stages`: Controls partial fine-tuning (e.g., set to `None` for full fine-tuning, or `0` for a frozen linear probe).
*   `tta_crops`: Number of deterministic crops averaged per test image for Test-Time Augmentation (TTA).

---

## Output Files

Upon completion, all outputs are saved in the `results/` and `checkpoints/` directories:
*   `checkpoints/best_multitask_equal.pth`: Best multi-task model checkpoint saved during training.
*   `checkpoints/final_multitask_model.pth`: Final trained model checkpoint.
*   `results/experiment_summary.json`: JSON dump containing configuration details and final evaluation accuracies on the held-out test set.
*   `results/*.png`: Confusion matrices, per-transformation breakdowns, cross-class traces, confidence distributions, and training curves.
