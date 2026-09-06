"""Mathematical Evaluation Script for Furniture Deduplication Benchmarks.

Calculates Pairwise Precision, Recall, F1 Score, and False Positive Analysis
against Ground Truth labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def evaluate_predictions(
    ground_truth_path: Path,
    predictions_csv_path: Path,
) -> dict[str, float | int]:
    """Evaluate duplicate detection results against ground truth.

    Args:
        ground_truth_path: Path to ground_truth.json.
        predictions_csv_path: Path to filtered_duplicates.csv or report.csv.

    Returns:
        Dictionary of metrics (Precision, Recall, F1, TP, FP, FN).
    """
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # 1. Build ground-truth pair set
    gt_pairs: set[tuple[str, str]] = set()
    for prod_id, members in gt_data["true_duplicate_clusters"].items():
        sorted_members = sorted(members)
        for i in range(len(sorted_members)):
            for j in range(i + 1, len(sorted_members)):
                gt_pairs.add((sorted_members[i], sorted_members[j]))

    negative_control_files = {nc["variant_file"] for nc in gt_data.get("negative_controls", [])}

    # 2. Build predicted pair set from report
    pred_pairs: set[tuple[str, str]] = set()
    pred_files_in_groups: set[str] = set()

    if predictions_csv_path.exists():
        df = pd.read_csv(predictions_csv_path)
        if not df.empty and "group_id" in df.columns:
            for gid, group_df in df.groupby("group_id"):
                members = sorted(group_df["file_name"].tolist())
                for i in range(len(members)):
                    pred_files_in_groups.add(members[i])
                    for j in range(i + 1, len(members)):
                        pred_pairs.add((members[i], members[j]))

    # 3. Calculate True Positives, False Positives, False Negatives
    tp_pairs = pred_pairs.intersection(gt_pairs)
    fp_pairs = pred_pairs - gt_pairs
    fn_pairs = gt_pairs - pred_pairs

    tp = len(tp_pairs)
    fp = len(fp_pairs)
    fn = len(fn_pairs)

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 100.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Negative control violations (did any color variant get wrongly grouped?)
    color_gate_violations = [f for f in negative_control_files if f in pred_files_in_groups]

    results = {
        "total_ground_truth_pairs": len(gt_pairs),
        "total_predicted_pairs": len(pred_pairs),
        "true_positives (TP)": tp,
        "false_positives (FP)": fp,
        "false_negatives (FN)": fn,
        "precision (%)": round(precision, 2),
        "recall (%)": round(recall, 2),
        "f1_score (%)": round(f1, 2),
        "negative_controls_tested": len(negative_control_files),
        "color_gate_violations": len(color_gate_violations),
        "target_precision_met (>95%)": precision >= 95.0,
    }

    print("\n" + "=" * 65)
    print("      MATHEMATICAL BENCHMARK EVALUATION SCORECARD")
    print("=" * 65)
    print(f"Ground Truth Duplicate Pairs: {len(gt_pairs)}")
    print(f"Predicted Duplicate Pairs:    {len(pred_pairs)}")
    print(f"True Positives  (TP):         {tp}")
    print(f"False Positives (FP):         {fp}")
    print(f"False Negatives (FN):         {fn}")
    print("-" * 65)
    print(f"PRECISION:                    {precision:.2f}%  (Target: >= 95.00%)")
    print(f"RECALL:                       {recall:.2f}%")
    print(f"F1-SCORE:                     {f1:.2f}%")
    print(f"Negative Controls Rejected:   {len(negative_control_files) - len(color_gate_violations)} / {len(negative_control_files)}")
    print(f"Color Gate Violations:        {len(color_gate_violations)}")
    print(f"STATUS:                       {'PASSED (>= 95% Precision)' if precision >= 95.0 else 'FAILED'}")
    print("=" * 65 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate furniture dedup predictions")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to ground_truth.json")
    parser.add_argument("--report", type=Path, required=True, help="Path to predictions CSV")
    args = parser.parse_args()

    evaluate_predictions(args.ground_truth, args.report)


if __name__ == "__main__":
    main()
