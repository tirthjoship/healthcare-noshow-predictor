"""Tests for adapters/ml/fairness.py — per-slice error rates and disparities."""

import numpy as np

from adapters.ml.fairness import fairness_report, group_metrics

_GROUP_KEYS = {"n", "base_rate", "selection_rate", "fpr", "fnr", "precision", "ece"}


class TestGroupMetrics:
    def test_returns_required_keys(self) -> None:
        y_true = np.array([1, 0, 1, 0], dtype=float)
        y_prob = np.array([0.8, 0.2, 0.6, 0.1])
        assert set(group_metrics(y_true, y_prob, threshold=0.5).keys()) == _GROUP_KEYS

    def test_perfect_separation(self) -> None:
        y_true = np.array([1, 1, 0, 0], dtype=float)
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = group_metrics(y_true, y_prob, threshold=0.5)
        assert m["fpr"] == 0.0
        assert m["fnr"] == 0.0
        assert m["precision"] == 1.0
        assert m["base_rate"] == 0.5
        assert m["selection_rate"] == 0.5

    def test_no_flags_gives_nan_precision_and_full_fnr(self) -> None:
        y_true = np.array([1, 1, 0, 0], dtype=float)
        y_prob = np.array([0.4, 0.3, 0.1, 0.2])
        m = group_metrics(y_true, y_prob, threshold=0.99)
        assert m["fpr"] == 0.0
        assert m["fnr"] == 1.0
        assert np.isnan(m["precision"])

    def test_all_flags_gives_full_fpr(self) -> None:
        y_true = np.array([1, 0, 0], dtype=float)
        y_prob = np.array([0.9, 0.8, 0.7])
        m = group_metrics(y_true, y_prob, threshold=0.5)
        assert m["fpr"] == 1.0
        assert m["fnr"] == 0.0


class TestFairnessReport:
    def _make_case(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        rng = np.random.default_rng(0)
        # Group F: negatives that the model over-flags (high FPR).
        # Group M: negatives the model correctly leaves low (low FPR).
        f_prob = rng.uniform(0.6, 0.9, size=150)
        m_prob = rng.uniform(0.0, 0.3, size=150)
        y_prob = np.concatenate([f_prob, m_prob])
        y_true = np.zeros(300, dtype=float)
        gender = np.array(["F"] * 150 + ["M"] * 150)
        return y_true, y_prob, {"gender": gender}

    def test_structure(self) -> None:
        y_true, y_prob, slices = self._make_case()
        report = fairness_report(y_true, y_prob, slices, threshold=0.5)
        assert report["n_total"] == 300
        assert set(report["overall"].keys()) == _GROUP_KEYS  # type: ignore[union-attr]
        assert "gender" in report["slices"]  # type: ignore[operator]
        assert set(report["slices"]["gender"].keys()) == {"F", "M"}  # type: ignore[index]

    def test_disparity_gap_detected(self) -> None:
        y_true, y_prob, slices = self._make_case()
        report = fairness_report(y_true, y_prob, slices, threshold=0.5)
        gap = report["disparities"]["gender"]["fpr_gap"]  # type: ignore[index]
        assert gap > 0.5  # F group heavily over-flagged vs M

    def test_small_groups_excluded_from_gap(self) -> None:
        y_true = np.zeros(210, dtype=float)
        y_prob = np.concatenate(
            [
                np.full(150, 0.9),  # group A (n=150, eligible)
                np.full(50, 0.9),  # group B (n=50, below min_group_size)
                np.full(10, 0.1),  # group C (n=10, below min_group_size)
            ]
        )
        labels = np.array(["A"] * 150 + ["B"] * 50 + ["C"] * 10)
        report = fairness_report(
            y_true, y_prob, {"grp": labels}, threshold=0.5, min_group_size=100
        )
        # Only group A meets min_group_size → fewer than two eligible → NaN gap.
        assert np.isnan(report["disparities"]["grp"]["fpr_gap"])  # type: ignore[index]
