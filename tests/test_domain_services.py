"""Skeleton tests for domain services (baseline risk flag)."""

from datetime import datetime

from domain.models import Patient, Encounter
from domain.services import baseline_readmission_risk_flag


def test_baseline_returns_one_of_three_categories() -> None:
    """baseline_readmission_risk_flag returns Low, Medium, or High Risk."""
    p = Patient(patient_id="P1", age=50, gender="M", insurance_type="Private")
    e = Encounter(
        encounter_id="E1",
        patient=p,
        encounter_date=datetime.now(),
        encounter_type="Elective",
        primary_diagnosis="Z00",
        comorbidity_count=0,
        length_of_stay_scheduled=1,
        prior_admissions_count=0,
    )
    result = baseline_readmission_risk_flag(e)
    assert result in ("Low Risk", "Medium Risk", "High Risk")
