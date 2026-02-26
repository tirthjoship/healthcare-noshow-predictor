"""Domain models for patient readmission risk.

Pure Python value objects. No pandas, numpy, or external ML imports.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .exceptions import InvalidAdmissionDataError, InvalidRiskAssessmentError


@dataclass(frozen=True)
class Patient:
    """Immutable value object representing a hospital patient.

    Attributes:
        patient_id: Unique identifier for the patient.
        age: Patient age in years (must be >= 0).
        gender: Patient gender ('M', 'F', 'Other').
        insurance_type: Insurance category.
    """

    patient_id: str
    age: int
    gender: str
    insurance_type: str

    def __post_init__(self) -> None:
        if self.age < 0:
            raise InvalidAdmissionDataError(f"Age must be non-negative, got {self.age}")
        if self.gender not in {"M", "F", "Other"}:
            raise InvalidAdmissionDataError(f"Invalid gender: {self.gender}")


@dataclass(frozen=True)
class Encounter:
    """Immutable value object representing a hospital encounter (admission).

    Pre-discharge features only; no post-discharge fields to avoid leakage.

    Attributes:
        encounter_id: Unique identifier for this encounter.
        patient: The patient.
        encounter_date: Date and time of encounter.
        encounter_type: Type (e.g. 'Emergency', 'Elective', 'Urgent').
        primary_diagnosis: ICD-style code for primary diagnosis.
        comorbidity_count: Number of documented comorbidities.
        length_of_stay_scheduled: Planned length of stay in days.
        prior_admissions_count: Admissions in past 12 months.
    """

    encounter_id: str
    patient: Patient
    encounter_date: datetime
    encounter_type: str
    primary_diagnosis: str
    comorbidity_count: int
    length_of_stay_scheduled: int
    prior_admissions_count: int

    def __post_init__(self) -> None:
        if self.encounter_type not in {"Emergency", "Elective", "Urgent"}:
            raise InvalidAdmissionDataError(
                f"Invalid encounter type: {self.encounter_type}"
            )
        if self.comorbidity_count < 0:
            raise InvalidAdmissionDataError("Comorbidity count must be non-negative")
        if self.length_of_stay_scheduled < 0:
            raise InvalidAdmissionDataError("Length of stay must be non-negative")


@dataclass(frozen=True)
class RiskOutcome:
    """Immutable value object representing a readmission risk prediction.

    Attributes:
        encounter_id: Reference to the encounter assessed.
        risk_score: Predicted probability of 30-day readmission (0.0 to 1.0).
        risk_category: Human-readable category.
        assessment_timestamp: When this assessment was generated.
        model_version: Version identifier of the model used.
        explanation: Optional feature importance (e.g. SHAP) for explainability.
    """

    encounter_id: str
    risk_score: float
    risk_category: str
    assessment_timestamp: datetime
    model_version: str
    explanation: Optional[dict[str, float]] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise InvalidRiskAssessmentError(
                f"Risk score must be in [0, 1], got {self.risk_score}"
            )
        if self.risk_category not in {"Low Risk", "Medium Risk", "High Risk"}:
            raise InvalidRiskAssessmentError(
                f"Invalid risk category: {self.risk_category}"
            )
