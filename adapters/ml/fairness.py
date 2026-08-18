"""Fairness reporting — per-slice error rates and calibration (ADR-009).

A ranked call list is only trustworthy if it does not systematically under- or
over-flag protected groups. This module computes, for each demographic slice,
the metrics that matter for an outreach use case:

- **base rate** — actual no-show rate in the slice (context, not a model metric)
- **selection rate** — fraction the model would flag at the operating threshold
- **FPR / FNR** — who gets called unnecessarily vs. missed
- **precision** — of those flagged, how many truly no-show
- **ECE** — is the probability honest *within* the slice (calibration can hide
  in the aggregate while failing per group)

All functions are pure: numpy in, plain dicts out. No plotting, no I/O.
"""

from __future__ import annotations

import numpy as np

from adapters.ml.evaluation import _expected_calibration_error


def group_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute error/selection/calibration metrics for one group.

    Args:
        y_true: Binary ground-truth labels (0/1) for the group.
        y_prob: Predicted no-show probabilities for the group.
        threshold: Decision threshold used to derive FPR/FNR/precision.
        n_bins: Bins for the Expected Calibration Error.

    Returns:
        Dict with n, base_rate, selection_rate, fpr, fnr, precision, ece.
        Rates with a zero denominator are reported as NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = int(len(y_true))
    y_pred = (y_prob >= threshold).astype(int)

    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    positives = tp + fn
    negatives = fp + tn
    flagged = tp + fp

    fpr = fp / negatives if negatives > 0 else float("nan")
    fnr = fn / positives if positives > 0 else float("nan")
    precision = tp / flagged if flagged > 0 else float("nan")
    base_rate = float(y_true.mean()) if n > 0 else float("nan")
    selection_rate = float(y_pred.mean()) if n > 0 else float("nan")
    ece = (
        _expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        if n > 0
        else float("nan")
    )

    return {
        "n": float(n),
        "base_rate": round(base_rate, 4),
        "selection_rate": round(selection_rate, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "precision": round(precision, 4),
        "ece": round(ece, 4),
    }


def _gap(values: list[float]) -> float:
    """Max-minus-min gap over finite values (NaN if fewer than two)."""
    finite = [v for v in values if not np.isnan(v)]
    if len(finite) < 2:
        return float("nan")
    return round(max(finite) - min(finite), 4)


def fairness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    slices: dict[str, np.ndarray],
    threshold: float,
    min_group_size: int = 100,
) -> dict[str, object]:
    """Build a per-slice fairness report with disparity gaps.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted no-show probabilities.
        slices: Map of attribute name → array of group labels (same length as
            y_true), e.g. {"gender": array(["F", "M", ...])}.
        threshold: Operating threshold for FPR/FNR/precision.
        min_group_size: Groups smaller than this are still reported but excluded
            from disparity-gap calculations (small-sample noise).

    Returns:
        Dict with threshold, overall metrics, per-slice group metrics, and
        FPR/FNR disparity gaps per attribute.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    report: dict[str, object] = {
        "threshold": round(float(threshold), 4),
        "n_total": int(len(y_true)),
        "overall": group_metrics(y_true, y_prob, threshold),
        "slices": {},
        "disparities": {},
    }
    slices_out: dict[str, object] = report["slices"]  # type: ignore[assignment]
    disparities_out: dict[str, object] = report["disparities"]  # type: ignore[assignment]

    for attr, labels in slices.items():
        labels = np.asarray(labels)
        groups: dict[str, dict[str, float]] = {}
        eligible_fpr: list[float] = []
        eligible_fnr: list[float] = []

        for value in sorted({str(v) for v in labels.tolist()}):
            mask = labels.astype(str) == value
            metrics = group_metrics(y_true[mask], y_prob[mask], threshold)
            groups[value] = metrics
            if metrics["n"] >= min_group_size:
                eligible_fpr.append(metrics["fpr"])
                eligible_fnr.append(metrics["fnr"])

        slices_out[attr] = groups
        disparities_out[attr] = {
            "fpr_gap": _gap(eligible_fpr),
            "fnr_gap": _gap(eligible_fnr),
        }

    return report
