"""CalibratedPredictor — XGBoost with isotonic calibration (project identity, ADR-008).

Wraps XGBClassifier in CalibratedClassifierCV(method="isotonic") to produce
well-calibrated probabilities. Calibrated probabilities are critical for the
clinic outreach use case: a "70% no-show risk" must mean ~70% of such patients
actually miss their appointment.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.logistic_predictor import _score_to_category
from domain.models import Appointment, NoShowOutcome


class CalibratedPredictor:
    """XGBoost predictor with isotonic calibration.

    Implements NoShowPredictorPort (structural duck-typing).

    Usage:
        predictor = CalibratedPredictor()
        predictor.train(appointments, outcomes)
        result = predictor.predict_no_show(appointment)
    """

    def __init__(self) -> None:
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._encoder: FeatureEncoder = FeatureEncoder()

    def train(self, appointments: list[Appointment], outcomes: list[bool]) -> None:
        """Fit XGBoost + isotonic calibration on training appointments.

        Uses cross-validation for calibration (ADR-008). Number of folds scales
        with dataset size to avoid single-sample folds on small fixtures.

        Args:
            appointments: Training appointment domain objects.
            outcomes: Corresponding no-show labels (True = no-show).
        """
        X = self._encoder.fit_transform(appointments, outcomes)
        y = np.array([int(o) for o in outcomes])
        base_model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            scale_pos_weight=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        n_folds = min(3, max(2, len(y) // 20))
        self._calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",
            cv=n_folds,
        )
        self._calibrated_model.fit(X, y)

    def predict_no_show(self, appointment: Appointment) -> NoShowOutcome:
        """Predict no-show risk for a single appointment.

        Args:
            appointment: The appointment to assess.

        Returns:
            NoShowOutcome with calibrated risk_score, risk_category, and metadata.

        Raises:
            RuntimeError: If train() has not been called first.
        """
        if self._calibrated_model is None:
            raise RuntimeError(
                "Model must be trained before prediction. Call train() first."
            )
        X = self._encoder.transform([appointment])
        prob = float(self._calibrated_model.predict_proba(X)[0, 1])
        return NoShowOutcome(
            appointment_id=appointment.appointment_id,
            risk_score=prob,
            risk_category=_score_to_category(prob),
            assessment_timestamp=datetime.now(),
            model_version="calibrated-xgboost-v1",
        )
