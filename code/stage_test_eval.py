"""
Stage: Evaluation on Test Set

Saves experiment summary, final model checkpoint, and reports total runtime.
This stage can be extended to load and evaluate on the held-out test set
once test metadata is prepared.
"""

import os
import json
import time
import torch

from config import CONFIG, RESULTS_DIR, CHECKPOINTS_DIR


def save_results(model_mt, start_time,
                 unimodal_bin_acc, unimodal_trans_acc,
                 multitask_bin_acc, multitask_trans_acc,
                 ablation_df, breakdown_df, trace_df):
    """
    Save all experiment results to JSON and the final model checkpoint.

    Args:
        model_mt: The trained multi-task model
        start_time: Timestamp when the experiment started
        unimodal_bin_acc: Unimodal binary baseline accuracy
        unimodal_trans_acc: Unimodal transform baseline accuracy
        multitask_bin_acc: Multi-task binary accuracy
        multitask_trans_acc: Multi-task transform accuracy
        ablation_df: DataFrame with ablation study results
        breakdown_df: DataFrame with per-transformation breakdown
        trace_df: DataFrame with cross-class trace analysis
    """
    print("\n" + "=" * 60)
    print("PHASE 9: SAVING RESULTS")
    print("=" * 60)

    # Save all results to a JSON summary
    final_summary = {
        "config": {k: str(v) for k, v in CONFIG.items()},
        "unimodal_binary_acc": float(unimodal_bin_acc),
        "unimodal_transform_acc": float(unimodal_trans_acc),
        "multitask_binary_acc": float(multitask_bin_acc),
        "multitask_transform_acc": float(multitask_trans_acc),
        "ablation_results": ablation_df.to_dict(orient="records") if ablation_df is not None else [],
        "per_transformation_breakdown": breakdown_df.to_dict(orient="records"),
        "cross_class_traces": trace_df.to_dict(orient="records"),
        "total_training_time_minutes": (time.time() - start_time) / 60,
    }

    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    # Save the final multi-task model
    final_model_path = os.path.join(CHECKPOINTS_DIR, "final_multitask_model.pth")
    torch.save(model_mt.state_dict(), final_model_path)

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE! Total time: {elapsed:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Model checkpoints: {CHECKPOINTS_DIR}/")
    print(f"Experiment summary: {summary_path}")
    print(f"{'='*60}")
