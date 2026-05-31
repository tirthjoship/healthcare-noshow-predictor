"""LogisticPredictor — logistic regression adapter implementing NoShowPredictorPort.

_score_to_category is module-level so XGBoostPredictor and CalibratedPredictor
can import it directly from this module (shared threshold logic).
"""

from __future__ import annotations

from datetime import datetime

from sklearn.linear_model import LogisticRegression

from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Appointment, NoShowOutcome


def _score_to_category(score: float) -> str:
    """Convert a raw probability to a human-readable risk category.

    Thresholds (ADR-locked):
        < 0.2  → Low Risk
        <= 0.4 → Medium Risk
        > 0.4  → High Risk
    """
    if score < 0.2:
        return "Low Risk"
    if score <= 0.4:
        return "Medium Risk"
    return "High Risk"


class LogisticPredictor:
    """Logistic regression adapter for no-show risk prediction.

    Implements NoShowPredictorPort (structural duck-typing — no explicit
    inheritance to keep domain free of adapter imports).

    Usage:
        predictor = LogisticPredictor()
        predictor.train(appointments, outcomes)
        result = predictor.predict_no_show(appointment)
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._encoder: FeatureEncoder = FeatureEncoder()

    def train(self, appointments: list[Appointment], outcomes: list[bool]) -> None:
        """Fit the logistic regression model on training appointments.

        Args:
            appointments: Training appointment domain objects.
            outcomes: Corresponding no-show labels (True = no-show).
        """
        X = self._encoder.fit_transform(appointments, outcomes)
        y = [int(o) for o in outcomes]
        self._model = LogisticRegression(max_iter=1000, random_state=42)
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
                "LogisticPredictor must be trained before calling predict_no_show(). "
                "Call train() first."
            )
        X = self._encoder.transform([appointment])
        prob = float(self._model.predict_proba(X)[0, 1])
        return NoShowOutcome(
            appointment_id=appointment.appointment_id,
            risk_score=prob,
            risk_category=_score_to_category(prob),
            assessment_timestamp=datetime.now(),
            model_version="logistic-v1",
        )
