"""ShapExplainer — SHAP feature attribution for no-show risk (ADR-012).

Explains the XGBoost model that sits underneath ``CalibratedPredictor``.
Isotonic calibration is a monotonic post-transform, so SHAP attributions on the
uncalibrated tree model describe the same risk *ordering* the outreach team acts
on when it ranks a daily call list.

This is the quantitative backing for ADR-012's framing: appointment no-shows are
an *operations* problem, not a *clinical* one. Scheduling-time features
(lead time, age) dominate; clinical comorbidities (hypertension, diabetes,
handicap) contribute little.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shap
from xgboost import XGBClassifier

from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Appointment


class ShapExplainer:
    """Fit an XGBoost model and expose SHAP feature attributions.

    Hyperparameters mirror ``CalibratedPredictor``'s base estimator so the
    explanations describe the same tree structure the production model calibrates.

    Usage:
        explainer = ShapExplainer()
        explainer.fit(train_appointments, train_labels)
        importance = explainer.global_importance(sample_appointments)
        contributions = explainer.explain_appointment(appointment)
    """

    def __init__(self) -> None:
        self._model: XGBClassifier | None = None
        self._encoder: FeatureEncoder = FeatureEncoder()
        self._explainer: Any | None = None

    def fit(self, appointments: list[Appointment], labels: list[bool]) -> None:
        """Fit the encoder + XGBoost model and build a TreeExplainer.

        Args:
            appointments: Training appointment domain objects.
            labels: Corresponding no-show labels (True = no-show).
        """
        X = self._encoder.fit_transform(appointments, labels)
        y = np.array([int(lb) for lb in labels])
        model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            scale_pos_weight=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X, y)
        self._model = model
        self._explainer = shap.TreeExplainer(model)

    def _require_fitted(self) -> Any:
        if self._model is None or self._explainer is None:
            raise RuntimeError(
                "ShapExplainer must be fitted before use. Call fit() first."
            )
        return self._explainer

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature-column names matching the SHAP value columns."""
        return self._encoder.get_feature_names()

    def encode(self, appointments: list[Appointment]) -> np.ndarray:
        """Return the encoded (N, 10) feature matrix for the given appointments."""
        return self._encoder.transform(appointments)

    def shap_values(self, appointments: list[Appointment]) -> np.ndarray:
        """Compute per-feature SHAP values (log-odds space) for appointments.

        Args:
            appointments: Appointments to explain.

        Returns:
            Float64 array of shape (len(appointments), 10).
        """
        explainer = self._require_fitted()
        X = self._encoder.transform(appointments)
        values = explainer.shap_values(X)
        return np.asarray(values, dtype=np.float64)

    def global_importance(self, appointments: list[Appointment]) -> dict[str, float]:
        """Mean absolute SHAP value per feature, sorted high → low.

        Args:
            appointments: Appointments to aggregate importance over.

        Returns:
            Ordered dict of feature name → mean |SHAP| (descending).
        """
        values = self.shap_values(appointments)
        mean_abs = np.abs(values).mean(axis=0)
        importance = {name: float(v) for name, v in zip(self.feature_names, mean_abs)}
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    def explain_appointment(self, appointment: Appointment) -> dict[str, float]:
        """Per-feature SHAP contributions for a single appointment.

        Suitable for populating ``NoShowOutcome.explanation``.

        Args:
            appointment: The appointment to explain.

        Returns:
            Dict of feature name → signed SHAP contribution.
        """
        values = self.shap_values([appointment])
        return {name: float(v) for name, v in zip(self.feature_names, values[0])}
