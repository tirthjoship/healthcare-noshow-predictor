"""Evaluation metrics adapter for no-show prediction models."""

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error over n_bins equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n_total = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)
    return float(ece)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    k: int = 20,
) -> dict[str, float | str]:
    """Compute AUC, F1 (optimal threshold), Brier score, precision@K, ECE.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        model_name: Identifier for the model being evaluated.
        k: Number of top predictions to use for precision@K.

    Returns:
        Dict with keys: model_name, auc, f1, brier, precision_at_k, ece.
        All numeric values rounded to 4 decimal places.
    """
    # AUC
    auc = float(roc_auc_score(y_true, y_prob))

    # F1 at optimal threshold (scan 0.10 → 0.90 step 0.01)
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_f1 = 0.0
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_f1:
            best_f1 = score

    # Brier score
    brier = float(brier_score_loss(y_true, y_prob))

    # Precision@K
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    precision_at_k = float(y_true[top_k_indices].mean())

    # ECE
    ece = _expected_calibration_error(y_true, y_prob)

    return {
        "model_name": model_name,
        "auc": round(auc, 4),
        "f1": round(best_f1, 4),
        "brier": round(brier, 4),
        "precision_at_k": round(precision_at_k, 4),
        "ece": round(ece, 4),
    }
