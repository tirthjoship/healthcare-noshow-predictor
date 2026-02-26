"""Skeleton tests for domain models (Patient, Encounter, RiskOutcome)."""

from datetime import datetime

import pytest

from domain.exceptions import InvalidAdmissionDataError, InvalidRiskAssessmentError
from domain.models import Patient, Encounter, RiskOutcome


def test_patient_valid_creation() -> None:
    """Valid Patient is created and immutable."""
    p = Patient(patient_id="P001", age=65, gender="M", insurance_type="Medicare")
    assert p.patient_id == "P001"
    assert p.age == 65


def test_patient_negative_age_raises() -> None:
    """Negative age raises InvalidAdmissionDataError."""
    with pytest.raises(InvalidAdmissionDataError):
        Patient(patient_id="P002", age=-1, gender="F", insurance_type="Private")


def test_encounter_valid_creation() -> None:
    """Valid Encounter is created."""
    p = Patient(patient_id="P001", age=70, gender="F", insurance_type="Medicaid")
    e = Encounter(
        encounter_id="E001",
        patient=p,
        encounter_date=datetime.now(),
        encounter_type="Emergency",
        primary_diagnosis="I50",
        comorbidity_count=2,
        length_of_stay_scheduled=5,
        prior_admissions_count=1,
    )
    assert e.encounter_id == "E001"
    assert e.encounter_type == "Emergency"


def test_risk_outcome_valid_creation() -> None:
    """Valid RiskOutcome is created."""
    r = RiskOutcome(
        encounter_id="E001",
        risk_score=0.5,
        risk_category="Medium Risk",
        assessment_timestamp=datetime.now(),
        model_version="v1.0",
    )
    assert r.risk_score == 0.5
    assert r.risk_category == "Medium Risk"


def test_risk_outcome_invalid_score_raises() -> None:
    """Risk score outside [0, 1] raises InvalidRiskAssessmentError."""
    with pytest.raises(InvalidRiskAssessmentError):
        RiskOutcome(
            encounter_id="E001",
            risk_score=1.5,
            risk_category="High Risk",
            assessment_timestamp=datetime.now(),
            model_version="v1.0",
        )
