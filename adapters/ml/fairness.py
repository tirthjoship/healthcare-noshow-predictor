"""Fairness reporting: per-slice error rates and calibration (ADR-009).

A ranked call list is only trustworthy if it does not systematically under- or
over-flag protected groups. This module computes, for each demographic slice,
the metrics that matter for an outreach use case:

- **base rate**: actual no-show rate in the slice (context, not a model metric)
- **selection rate**: fraction the model would flag at the operating threshold
- **FPR / FNR**: who gets called unnecessarily vs. missed
- **precision**: of those flagged, how many truly no-show
- **ECE**: is the probability honest *within* the slice (calibration can hide
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


def _select_top_k_per_day(y_prob: np.ndarray, days: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the K highest-risk rows within each day."""
    selected = np.zeros(len(y_prob), dtype=bool)
    for day in np.unique(days):
        idx = np.flatnonzero(days == day)
        top = idx[np.argsort(-y_prob[idx], kind="stable")[: max(0, k)]]
        selected[top] = True
    return selected


def capacity_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    days: np.ndarray,
    slices: dict[str, np.ndarray],
    k: int,
) -> dict[str, object]:
    """Fairness at the call-list operating point: top-K per day (ADR-005, ADR-013).

    A fixed probability threshold describes a screening rule the app never uses.
    The outreach team calls the K riskiest patients each day, so this report asks,
    per group: what share of the group's no-shows make the call list, and is the
    group's share of the list in line with its share of appointments and no-shows?

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted no-show probabilities.
        days: Appointment day per row; selection happens within each day.
        slices: Map of attribute name → array of group labels.
        k: Daily call capacity.

    Returns:
        Dict with k, n_selected, overall precision/capture, and per-slice groups
        with no_shows_captured, share_of_list, share_of_appointments, share_of_no_shows.
        Ties at the daily cutoff are broken by row order (stable sort); isotonic
        calibration produces tied scores, so exact precision can shift slightly.
        Capture for a group with no no-shows is NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    days = np.asarray(days).astype(str)
    selected = _select_top_k_per_day(y_prob, days, k)

    n_total = len(y_true)
    n_selected = int(selected.sum())
    total_no_shows = float(y_true.sum())
    hits = float((selected & (y_true == 1)).sum())

    report: dict[str, object] = {
        "k": int(k),
        "n_days": int(len(np.unique(days))),
        "n_selected": n_selected,
        "overall": {
            "precision": round(hits / n_selected, 4) if n_selected else float("nan"),
            "no_shows_captured": (
                round(hits / total_no_shows, 4) if total_no_shows else float("nan")
            ),
        },
        "slices": {},
    }
    slices_out: dict[str, object] = report["slices"]  # type: ignore[assignment]

    for attr, labels in slices.items():
        labels = np.asarray(labels).astype(str)
        groups: dict[str, dict[str, float]] = {}
        for value in sorted(set(labels.tolist())):
            mask = labels == value
            group_no_shows = float(y_true[mask].sum())
            group_hits = float((selected & mask & (y_true == 1)).sum())
            groups[value] = {
                "n": float(mask.sum()),
                "no_shows_captured": (
                    round(group_hits / group_no_shows, 4)
                    if group_no_shows
                    else float("nan")
                ),
                "share_of_list": (
                    round(float((selected & mask).sum()) / n_selected, 4)
                    if n_selected
                    else float("nan")
                ),
                "share_of_appointments": (
                    round(float(mask.sum()) / n_total, 4) if n_total else float("nan")
                ),
                "share_of_no_shows": (
                    round(group_no_shows / total_no_shows, 4)
                    if total_no_shows
                    else float("nan")
                ),
            }
        slices_out[attr] = groups

    return report
