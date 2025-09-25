from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def _to_numpy(array_like: Iterable[Any]) -> np.ndarray:
    if isinstance(array_like, (pd.Series, pd.Index)):
        return array_like.to_numpy()
    if isinstance(array_like, pd.DataFrame):
        return array_like.squeeze().to_numpy()
    return np.asarray(array_like)


def compute_fairness_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    sensitive: Iterable[Any],
) -> Dict[str, Any]:
    """
    Compute simple group fairness metrics for binary predictions.

    Returns a dictionary with:
    - overall: overall positive rate
    - by_group: DataFrame with per-group rates and differences
    - disparate_impact_min_ratio: min(selection_rate_group / selection_rate_overall)
    - demographic_parity_difference: max|selection_rate_group - selection_rate_overall|
    """

    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    sens_np = _to_numpy(sensitive)

    if y_pred_np.ndim != 1:
        y_pred_np = y_pred_np.ravel()
    if y_true_np.ndim != 1:
        y_true_np = y_true_np.ravel()
    if sens_np.ndim != 1:
        sens_np = sens_np.ravel()

    if len(y_pred_np) != len(sens_np):
        raise ValueError("Length mismatch: y_pred and sensitive must have same length")

    df = pd.DataFrame({
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "sensitive": sens_np,
    })

    # Ensure binary values 0/1
    df["y_pred"] = (df["y_pred"].astype(float) >= 0.5).astype(int)
    df["y_true"] = (df["y_true"].astype(float) >= 0.5).astype(int)

    overall_rate = float(df["y_pred"].mean())

    group_stats = (
        df.groupby("sensitive")
        .agg(
            count=("y_pred", "size"),
            selection_rate=("y_pred", "mean"),
            tpr=(lambda x: np.nan),
            fpr=(lambda x: np.nan),
        )
        .reset_index()
    )

    # Compute simple TPR/FPR per group if ground truth is provided (not all NaN)
    if df["y_true"].notna().any():
        by_group = []
        for group, sub in df.groupby("sensitive"):
            positives = (sub["y_true"] == 1)
            negatives = (sub["y_true"] == 0)
            tpr = float(((sub["y_pred"] == 1) & positives).sum() / positives.sum()) if positives.any() else np.nan
            fpr = float(((sub["y_pred"] == 1) & negatives).sum() / negatives.sum()) if negatives.any() else np.nan
            by_group.append({
                "sensitive": group,
                "count": int(len(sub)),
                "selection_rate": float(sub["y_pred"].mean()),
                "tpr": tpr,
                "fpr": fpr,
            })
        group_stats = pd.DataFrame(by_group)

    group_stats["diff_from_overall"] = group_stats["selection_rate"] - overall_rate

    # Disparate impact (selection rate ratio) relative to overall
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = group_stats["selection_rate"].to_numpy() / (overall_rate if overall_rate > 0 else np.nan)
    disparate_impact_min_ratio = float(np.nanmin(ratios)) if ratios.size else np.nan
    demographic_parity_difference = float(np.nanmax(np.abs(group_stats["diff_from_overall"])) if len(group_stats) else np.nan)

    return {
        "overall": overall_rate,
        "by_group": group_stats.sort_values("sensitive").reset_index(drop=True),
        "disparate_impact_min_ratio": disparate_impact_min_ratio,
        "demographic_parity_difference": demographic_parity_difference,
    }


