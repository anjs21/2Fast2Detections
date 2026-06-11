"""
Stage: Multi-Task Joint Training

Trains the shared-backbone multi-task model with joint loss,
evaluates it, and compares against unimodal baselines.
"""

from config import CONFIG
from models import MultiTaskModel
from training import train_multitask_model
from evaluation import (
    collect_multitask_predictions,
    full_classification_report,
    per_transformation_breakdown,
    cross_class_trace_analysis,
    plot_training_curves,
    run_visual_inference,
    print_comparative_summary,
)


def run_multimodal_training(mt_train_loader, mt_val_loader, val_df, val_tfm):
    """
    Train and evaluate the multi-task joint model.

    Returns:
        model_mt: Trained multi-task model
        results: Prediction results dict
        multitask_bin_acc: Validation binary accuracy
        multitask_trans_acc: Validation transform accuracy
    """
    print("\n" + "=" * 60)
    print("PHASE 3: MULTI-TASK JOINT TRAINING")
    print("=" * 60)

    model_mt = MultiTaskModel(
        backbone_name=CONFIG["backbone"],
        trainable_backbone_stages=CONFIG["trainable_backbone_stages"],
    ).to(CONFIG["device"])
    model_mt, logger_mt = train_multitask_model(
        model_mt, mt_train_loader, mt_val_loader, CONFIG, w1=0.5, w2=0.5, tag="multitask_equal"
    )
    plot_training_curves(logger_mt, "Multi-Task Joint Training (w1=0.5, w2=0.5)",
                         "curves_multitask.png")

    # Full evaluation
    results = collect_multitask_predictions(model_mt, mt_val_loader, CONFIG["device"])

    full_classification_report(results["preds_bin"], results["labels_bin"],
                               ["Real", "Fake"], "Multi-Task Real/Fake",
                               probs=results["probs_bin"])
    full_classification_report(results["preds_trans"], results["labels_trans"],
                               ["Original", "Transmitted", "Redigitalized"],
                               "Multi-Task Transformation")

    multitask_bin_acc = (results["preds_bin"] == results["labels_bin"]).mean()
    multitask_trans_acc = (results["preds_trans"] == results["labels_trans"]).mean()

    return model_mt, results, multitask_bin_acc, multitask_trans_acc


def run_comparison_and_analysis(results, val_df, val_tfm, model_mt,
                                 unimodal_bin_acc, unimodal_trans_acc,
                                 multitask_bin_acc, multitask_trans_acc):
    """
    Run unimodal vs multi-task comparison, per-transformation breakdown,
    cross-class analysis, and visual inference.
    """
    # ---- Comparison ----
    print("\n" + "=" * 60)
    print("PHASE 4: UNIMODAL vs MULTI-TASK COMPARISON")
    print("=" * 60)

    comparison_df = print_comparative_summary(
        unimodal_bin_acc, unimodal_trans_acc,
        multitask_bin_acc, multitask_trans_acc
    )

    # ---- Per-Transformation Breakdown ----
    print("\n" + "=" * 60)
    print("PHASE 5: PER-TRANSFORMATION ACCURACY BREAKDOWN")
    print("=" * 60)

    breakdown_df = per_transformation_breakdown(results, val_df)

    # ---- Cross-Class Trace Analysis ----
    print("\n" + "=" * 60)
    print("PHASE 6: CROSS-CLASS TRANSFORMATION TRACE ANALYSIS")
    print("=" * 60)

    trace_df = cross_class_trace_analysis(results)

    # ---- Visual Inference ----
    print("\n" + "=" * 60)
    print("PHASE 8: VISUAL INFERENCE")
    print("=" * 60)

    run_visual_inference(model_mt, val_df, val_tfm, CONFIG["device"], num_samples=6)

    return comparison_df, breakdown_df, trace_df
