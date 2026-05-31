"""Tests for adapters/ml/evaluation.py — evaluate_model metrics."""

import numpy as np
import pytest

from adapters.ml.evaluation import evaluate_model

REQUIRED_KEYS = {"model_name", "auc", "f1", "brier", "precision_at_k", "ece"}


def test_returns_required_keys() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=50).astype(float)
    y_prob = rng.uniform(0, 1, size=50)
    result = evaluate_model(y_true, y_prob, model_name="test", k=10)
    assert set(result.keys()) == REQUIRED_KEYS


def test_perfect_predictions() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    y_prob = np.array([0.99, 0.98, 0.97, 0.01, 0.02, 0.03])
    result = evaluate_model(y_true, y_prob, model_name="perfect", k=3)
    assert result["auc"] == 1.0
    assert result["brier"] == 0.0 or result["brier"] < 0.01
    assert result["precision_at_k"] == 1.0


def test_auc_bounded() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100).astype(float)
    y_prob = rng.uniform(0, 1, size=100)
    result = evaluate_model(y_true, y_prob, model_name="random", k=20)
    for key in ("auc", "f1", "brier", "precision_at_k", "ece"):
        assert 0.0 <= float(result[key]) <= 1.0, f"{key} out of [0,1]: {result[key]}"


def test_precision_at_k() -> None:
    # Top-2 by probability are both positive → precision@2 = 1.0
    y_true = np.array([1, 0, 1, 0, 0], dtype=float)
    y_prob = np.array([0.9, 0.3, 0.8, 0.2, 0.1])
    result = evaluate_model(y_true, y_prob, model_name="prec_test", k=2)
    assert result["precision_at_k"] == 1.0


def test_model_name_in_result() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=30).astype(float)
    y_prob = rng.uniform(0, 1, size=30)
    result = evaluate_model(y_true, y_prob, model_name="xgboost_v1", k=5)
    assert result["model_name"] == "xgboost_v1"
