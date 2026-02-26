"""Port interfaces (protocols) for patient readmission risk.

Adapters implement these ports; domain and application depend only
on these abstractions. Ports are designed to support data leakage pruning.
"""

from datetime import datetime
from typing import Protocol

from .models import Encounter, RiskOutcome


class PatientDataRepository(Protocol):
    """Port: load patient/encounter data with leakage prevention.

    Implementations must enforce exclusion of post-discharge features
    and raise DataLeakageError if such features are present.
    """

    def get_encounters(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Encounter]:
        """Load encounter records within the specified date range.

        Args:
            start_date: Optional filter for encounters on or after this date.
            end_date: Optional filter for encounters before this date.

        Returns:
            List of Encounter objects with only pre-discharge features.

        Raises:
            DataLeakageError: If post-discharge features are detected.
            InvalidAdmissionDataError: If data fails validation.
        """
        ...


class RiskPredictorPort(Protocol):
    """Port: ML model that predicts readmission risk.

    Implementations must use only pre-discharge features from Encounter.
    """

    def predict_readmission_risk(self, encounter: Encounter) -> RiskOutcome:
        """Predict 30-day readmission risk for a given encounter.

        Args:
            encounter: Encounter with pre-discharge features only.

        Returns:
            RiskOutcome with score, category, and optional explanation.
        """
        ...

    def train(
        self, encounters: list[Encounter], outcomes: list[bool]
    ) -> None:
        """Train the model on historical encounters and outcomes.

        Args:
            encounters: Historical encounters (pre-discharge features only).
            outcomes: Boolean outcomes (True = readmitted within 30 days).

        Raises:
            DataLeakageError: If training data contains post-discharge features.
        """
        ...
