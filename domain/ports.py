"""Port interfaces (protocols) for appointment no-show prediction.

Adapters implement these ports; domain and application depend only
on these abstractions. Ports enforce leakage prevention — all features
must be knowable at scheduling time.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

from .models import Appointment, NoShowOutcome


@runtime_checkable
class AppointmentRepository(Protocol):
    """Port: load appointment data with leakage prevention.

    Implementations must ensure all features are knowable at scheduling time.
    Post-appointment data (e.g. actual attendance) must not appear in features.
    """

    def get_appointments(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Appointment]:
        """Load appointment records within the specified date range.

        Args:
            start_date: Optional filter for appointments on or after this date.
            end_date: Optional filter for appointments before this date.

        Returns:
            List of Appointment objects with only pre-appointment features.

        Raises:
            DataLeakageError: If post-appointment features are detected.
            InvalidAppointmentDataError: If data fails validation.
        """
        ...

    def get_labels(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[bool]:
        """Load no-show labels aligned with get_appointments.

        Returns:
            List of bools (True = no-show) in same order as get_appointments.
        """
        ...


@runtime_checkable
class NoShowPredictorPort(Protocol):
    """Port: ML model that predicts appointment no-show risk.

    Implementations must use only pre-appointment features from Appointment.
    """

    def predict_no_show(self, appointment: Appointment) -> NoShowOutcome:
        """Predict no-show risk for a given appointment.

        Args:
            appointment: Appointment with pre-appointment features only.

        Returns:
            NoShowOutcome with score, category, and optional explanation.
        """
        ...

    def train(
        self, appointments: list[Appointment], outcomes: list[bool]
    ) -> None:
        """Train the model on historical appointments and outcomes.

        Args:
            appointments: Historical appointments (pre-appointment features).
            outcomes: Boolean outcomes (True = patient did not show up).

        Raises:
            DataLeakageError: If training data contains post-appointment features.
        """
        ...
