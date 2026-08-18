"""Tests for adapters/ml/shap_explainer.py — SHAP feature attribution."""

from datetime import datetime

import numpy as np
import pytest

from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.shap_explainer import ShapExplainer
from domain.models import Appointment, Patient

_SCHEDULED = datetime(2016, 1, 1, 8, 0, 0)
_APPOINTMENT = datetime(2016, 1, 10, 8, 0, 0)
_FEATURES = set(FeatureEncoder.FEATURE_NAMES)


def _make_appointment(i: int, no_show: bool) -> Appointment:
    patient = Patient(
        patient_id=f"P{i}", age=20 + (i % 60), gender="F" if i % 2 else "M"
    )
    return Appointment(
        appointment_id=f"A{i}",
        patient=patient,
        scheduled_day=_SCHEDULED,
        appointment_day=_APPOINTMENT,
        neighbourhood=["ALPHA", "BETA", "GAMMA"][i % 3],
        scholarship=i % 2,
        hypertension=1 if i % 4 == 0 else 0,
        diabetes=1 if i % 5 == 0 else 0,
        alcoholism=0,
        handicap=0,
        sms_received=1 if no_show else 0,
        lead_time_days=i % 40,
    )


def _training_data(n: int = 60) -> tuple[list[Appointment], list[bool]]:
    appointments = [_make_appointment(i, i % 3 == 0) for i in range(n)]
    labels = [i % 3 == 0 for i in range(n)]
    return appointments, labels


class TestShapExplainer:
    def test_global_importance_covers_all_features(self) -> None:
        appts, labels = _training_data()
        explainer = ShapExplainer()
        explainer.fit(appts, labels)
        importance = explainer.global_importance(appts)
        assert set(importance.keys()) == _FEATURES
        assert all(v >= 0.0 for v in importance.values())

    def test_global_importance_sorted_descending(self) -> None:
        appts, labels = _training_data()
        explainer = ShapExplainer()
        explainer.fit(appts, labels)
        values = list(explainer.global_importance(appts).values())
        assert values == sorted(values, reverse=True)

    def test_shap_values_shape(self) -> None:
        appts, labels = _training_data()
        explainer = ShapExplainer()
        explainer.fit(appts, labels)
        values = explainer.shap_values(appts[:5])
        assert values.shape == (5, len(FeatureEncoder.FEATURE_NAMES))

    def test_explain_appointment_returns_all_features(self) -> None:
        appts, labels = _training_data()
        explainer = ShapExplainer()
        explainer.fit(appts, labels)
        contributions = explainer.explain_appointment(appts[0])
        assert set(contributions.keys()) == _FEATURES

    def test_feature_names_property(self) -> None:
        explainer = ShapExplainer()
        assert explainer.feature_names == FeatureEncoder.FEATURE_NAMES

    def test_use_before_fit_raises(self) -> None:
        explainer = ShapExplainer()
        appts, _ = _training_data(3)
        with pytest.raises(RuntimeError):
            explainer.shap_values(appts)

    def test_encode_shape(self) -> None:
        appts, labels = _training_data()
        explainer = ShapExplainer()
        explainer.fit(appts, labels)
        matrix = explainer.encode(appts[:4])
        assert matrix.shape == (4, len(FeatureEncoder.FEATURE_NAMES))
        assert np.isfinite(matrix).all()
