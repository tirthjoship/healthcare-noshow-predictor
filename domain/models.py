"""Domain models for appointment no-show prediction.

Pure Python value objects. No pandas, numpy, or external ML imports.
All features must be knowable at scheduling time (leakage prevention).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .exceptions import InvalidAppointmentDataError, InvalidNoShowPredictionError


@dataclass(frozen=True)
class Patient:
    """Immutable value object representing a clinic patient.

    Attributes:
        patient_id: Unique identifier (may have multiple appointments).
        age: Patient age in years (must be >= 0).
        gender: Patient gender ('M' or 'F').
    """

    patient_id: str
    age: int
    gender: str

    def __post_init__(self) -> None:
        if self.age < 0:
            raise InvalidAppointmentDataError(
                f"Age must be non-negative, got {self.age}"
            )
        if self.gender not in {"M", "F"}:
            raise InvalidAppointmentDataError(f"Invalid gender: {self.gender}")


@dataclass(frozen=True)
class Appointment:
    """Immutable value object representing a medical appointment.

    All fields are knowable at scheduling time — no post-appointment data.

    Attributes:
        appointment_id: Unique identifier for this appointment.
        patient: The patient who booked.
        scheduled_day: When the appointment was scheduled (booking timestamp).
        appointment_day: When the appointment is set for.
        neighbourhood: Clinic neighbourhood (high cardinality).
        scholarship: Whether patient is enrolled in Bolsa Família (0/1).
        hypertension: Whether patient has hypertension (0/1).
        diabetes: Whether patient has diabetes (0/1).
        alcoholism: Whether patient has alcoholism (0/1).
        handicap: Handicap level (0-4).
        sms_received: Whether patient received SMS reminder (0/1).
            NOTE: This is an intervention feature — document confound.
        lead_time_days: Days between scheduling and appointment (derived).
    """

    appointment_id: str
    patient: Patient
    scheduled_day: datetime
    appointment_day: datetime
    neighbourhood: str
    scholarship: int
    hypertension: int
    diabetes: int
    alcoholism: int
    handicap: int
    sms_received: int
    lead_time_days: int

    def __post_init__(self) -> None:
        if self.lead_time_days < 0:
            raise InvalidAppointmentDataError(
                f"Lead time must be non-negative, got {self.lead_time_days}"
            )
        for field_name in ("scholarship", "hypertension", "diabetes", "alcoholism"):
            val = getattr(self, field_name)
            if val not in {0, 1}:
                raise InvalidAppointmentDataError(
                    f"{field_name} must be 0 or 1, got {val}"
                )
        if not 0 <= self.handicap <= 4:
            raise InvalidAppointmentDataError(
                f"Handicap must be 0-4, got {self.handicap}"
            )
        if self.sms_received not in {0, 1}:
            raise InvalidAppointmentDataError(
                f"sms_received must be 0 or 1, got {self.sms_received}"
            )


@dataclass(frozen=True)
class NoShowOutcome:
    """Immutable value object representing a no-show risk prediction.

    Attributes:
        appointment_id: Reference to the appointment assessed.
        risk_score: Predicted probability of no-show (0.0 to 1.0).
        risk_category: Human-readable category.
        assessment_timestamp: When this assessment was generated.
        model_version: Version identifier of the model used.
        explanation: Optional feature importance (e.g. SHAP values).
    """

    appointment_id: str
    risk_score: float
    risk_category: str
    assessment_timestamp: datetime
    model_version: str
    explanation: Optional[dict[str, float]] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise InvalidNoShowPredictionError(
                f"Risk score must be in [0, 1], got {self.risk_score}"
            )
        if self.risk_category not in {"Low Risk", "Medium Risk", "High Risk"}:
            raise InvalidNoShowPredictionError(
                f"Invalid risk category: {self.risk_category}"
            )
