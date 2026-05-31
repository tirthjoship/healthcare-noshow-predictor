"""XGBoostPredictor — gradient-boosted tree adapter implementing NoShowPredictorPort.

Uses XGBClassifier with scale_pos_weight to handle the ~20% no-show class imbalance
(ADR-locked: scale_pos_weight=4 approximates 80/20 majority/minority ratio).
"""

from __future__ import annotations

from datetime import datetime

from xgboost import XGBClassifier

from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.logistic_predictor import _score_to_category
from domain.models import Appointment, NoShowOutcome


class XGBoostPredictor:
    """XGBoost gradient-boosted tree adapter for no-show risk prediction.

    Implements NoShowPredictorPort (structural duck-typing — no explicit
    inheritance to keep domain free of adapter imports).

    Usage:
        predictor = XGBoostPredictor()
        predictor.train(appointments, outcomes)
        result = predictor.predict_no_show(appointment)
    """

    def __init__(self) -> None:
        self._model: XGBClassifier | None = None
        self._encoder: FeatureEncoder = FeatureEncoder()

    def train(self, appointments: list[Appointment], outcomes: list[bool]) -> None:
        """Fit the XGBoost model on training appointments.

        Args:
            appointments: Training appointment domain objects.
            outcomes: Corresponding no-show labels (True = no-show).
        """
        X = self._encoder.fit_transform(appointments, outcomes)
        y = [int(o) for o in outcomes]
        self._model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            scale_pos_weight=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)

    def predict_no_show(self, appointment: Appointment) -> NoShowOutcome:
        """Predict no-show risk for a single appointment.

        Args:
            appointment: The appointment to assess.

        Returns:
            NoShowOutcome with risk_score, risk_category, and model metadata.

        Raises:
            RuntimeError: If train() has not been called first.
        """
        if self._model is None:
            raise RuntimeError(
                "XGBoostPredictor must be trained before calling predict_no_show(). "
                "Call train() first."
            )
        X = self._encoder.transform([appointment])
        prob = float(self._model.predict_proba(X)[0, 1])
        return NoShowOutcome(
            appointment_id=appointment.appointment_id,
            risk_score=prob,
            risk_category=_score_to_category(prob),
            assessment_timestamp=datetime.now(),
            model_version="xgboost-v1",
        )
