"""Tests for domain services (baseline no-show risk flag)."""

from datetime import datetime

from domain.models import Appointment, Patient
from domain.services import baseline_no_show_risk_flag


def _make_appointment(**overrides: object) -> Appointment:
    patient = Patient(patient_id="P1", age=50, gender="M")
    defaults = {
        "appointment_id": "A1",
        "patient": patient,
        "scheduled_day": datetime(2016, 4, 29),
        "appointment_day": datetime(2016, 5, 1),
        "neighbourhood": "CENTRO",
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        "lead_time_days": 2,
    }
    defaults.update(overrides)
    return Appointment(**defaults)  # type: ignore[arg-type]


class TestBaselineNoShowRisk:
    def test_returns_valid_category(self) -> None:
        result = baseline_no_show_risk_flag(_make_appointment())
        assert result in ("Low Risk", "Medium Risk", "High Risk")

    def test_low_risk_same_day_older_patient(self) -> None:
        appt = _make_appointment(
            patient=Patient(patient_id="P2", age=60, gender="F"),
            lead_time_days=0,
            sms_received=0,
            scholarship=0,
        )
        assert baseline_no_show_risk_flag(appt) == "Low Risk"

    def test_high_risk_young_long_lead_sms(self) -> None:
        appt = _make_appointment(
            patient=Patient(patient_id="P3", age=25, gender="M"),
            lead_time_days=35,
            sms_received=1,
            scholarship=1,
        )
        assert baseline_no_show_risk_flag(appt) == "High Risk"

    def test_medium_risk_one_factor(self) -> None:
        appt = _make_appointment(
            patient=Patient(patient_id="P4", age=60, gender="F"),
            lead_time_days=20,
            sms_received=0,
            scholarship=0,
        )
        assert baseline_no_show_risk_flag(appt) == "Medium Risk"
