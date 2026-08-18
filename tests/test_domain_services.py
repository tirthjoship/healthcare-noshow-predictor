"""Tests for domain services (baseline no-show risk flag + business impact)."""

from datetime import datetime

import pytest

from domain.models import Appointment, Patient
from domain.services import baseline_no_show_risk_flag, estimate_business_impact


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


class TestEstimateBusinessImpact:
    def test_known_calculation(self) -> None:
        result = estimate_business_impact(
            precision_at_k=0.5,
            daily_call_capacity=20,
            cost_per_noshow=200.0,
            working_days_per_month=22,
        )
        assert result["recoverable_slots_per_day"] == 10.0
        assert result["recoverable_slots_per_month"] == 220.0
        assert result["monthly_value"] == 44000.0
        assert result["annual_value"] == 528000.0

    def test_zero_precision_yields_zero_value(self) -> None:
        result = estimate_business_impact(
            precision_at_k=0.0,
            daily_call_capacity=30,
            cost_per_noshow=200.0,
        )
        assert result["monthly_value"] == 0.0

    def test_inputs_echoed_back(self) -> None:
        result = estimate_business_impact(
            precision_at_k=0.45,
            daily_call_capacity=25,
            cost_per_noshow=200.0,
        )
        assert result["precision_at_k"] == 0.45
        assert result["daily_call_capacity"] == 25.0
        assert result["cost_per_noshow"] == 200.0
        assert result["working_days_per_month"] == 22.0

    def test_precision_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_business_impact(
                precision_at_k=1.5, daily_call_capacity=20, cost_per_noshow=200.0
            )

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_business_impact(
                precision_at_k=0.5, daily_call_capacity=-1, cost_per_noshow=200.0
            )
