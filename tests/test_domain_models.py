"""Tests for domain models (Patient, Appointment, NoShowOutcome)."""

from datetime import datetime

import pytest

from domain.exceptions import InvalidAppointmentDataError, InvalidNoShowPredictionError
from domain.models import Appointment, NoShowOutcome, Patient


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_patient(**overrides: object) -> Patient:
    defaults = {"patient_id": "P001", "age": 45, "gender": "F"}
    defaults.update(overrides)
    return Patient(**defaults)  # type: ignore[arg-type]


def _make_appointment(**overrides: object) -> Appointment:
    defaults = {
        "appointment_id": "A001",
        "patient": _make_patient(),
        "scheduled_day": datetime(2016, 4, 29, 18, 0),
        "appointment_day": datetime(2016, 5, 5),
        "neighbourhood": "JARDIM CAMBURI",
        "scholarship": 0,
        "hypertension": 1,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        "lead_time_days": 6,
    }
    defaults.update(overrides)
    return Appointment(**defaults)  # type: ignore[arg-type]


# ── Patient ──────────────────────────────────────────────────────────


class TestPatient:
    def test_valid_creation(self) -> None:
        p = _make_patient()
        assert p.patient_id == "P001"
        assert p.age == 45
        assert p.gender == "F"

    def test_negative_age_raises(self) -> None:
        with pytest.raises(InvalidAppointmentDataError, match="non-negative"):
            _make_patient(age=-1)

    def test_invalid_gender_raises(self) -> None:
        with pytest.raises(InvalidAppointmentDataError, match="Invalid gender"):
            _make_patient(gender="X")

    def test_immutable(self) -> None:
        p = _make_patient()
        with pytest.raises(AttributeError):
            p.age = 50  # type: ignore[misc]


# ── Appointment ──────────────────────────────────────────────────────


class TestAppointment:
    def test_valid_creation(self) -> None:
        a = _make_appointment()
        assert a.appointment_id == "A001"
        assert a.lead_time_days == 6

    def test_negative_lead_time_raises(self) -> None:
        with pytest.raises(InvalidAppointmentDataError, match="Lead time"):
            _make_appointment(lead_time_days=-1)

    def test_invalid_binary_field_raises(self) -> None:
        with pytest.raises(InvalidAppointmentDataError, match="scholarship"):
            _make_appointment(scholarship=2)

    def test_invalid_handicap_raises(self) -> None:
        with pytest.raises(InvalidAppointmentDataError, match="Handicap"):
            _make_appointment(handicap=5)

    def test_invalid_sms_raises(self) -> None:
        with pytest.raises(InvalidAppointmentDataError, match="sms_received"):
            _make_appointment(sms_received=3)

    def test_immutable(self) -> None:
        a = _make_appointment()
        with pytest.raises(AttributeError):
            a.lead_time_days = 10  # type: ignore[misc]


# ── NoShowOutcome ────────────────────────────────────────────────────


class TestNoShowOutcome:
    def test_valid_creation(self) -> None:
        r = NoShowOutcome(
            appointment_id="A001",
            risk_score=0.35,
            risk_category="Medium Risk",
            assessment_timestamp=datetime.now(),
            model_version="v1.0",
        )
        assert r.risk_score == 0.35

    def test_score_too_high_raises(self) -> None:
        with pytest.raises(InvalidNoShowPredictionError):
            NoShowOutcome(
                appointment_id="A001",
                risk_score=1.5,
                risk_category="High Risk",
                assessment_timestamp=datetime.now(),
                model_version="v1.0",
            )

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(InvalidNoShowPredictionError):
            NoShowOutcome(
                appointment_id="A001",
                risk_score=0.5,
                risk_category="Very High Risk",
                assessment_timestamp=datetime.now(),
                model_version="v1.0",
            )

    def test_with_explanation(self) -> None:
        r = NoShowOutcome(
            appointment_id="A001",
            risk_score=0.7,
            risk_category="High Risk",
            assessment_timestamp=datetime.now(),
            model_version="v1.0",
            explanation={"lead_time_days": 0.35, "age": -0.12},
        )
        assert r.explanation is not None
        assert "lead_time_days" in r.explanation
