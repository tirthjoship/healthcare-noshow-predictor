"""Property-based tests for domain model invariants using Hypothesis."""

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from domain.exceptions import InvalidAppointmentDataError, InvalidNoShowPredictionError
from domain.models import Appointment, NoShowOutcome, Patient
from domain.services import baseline_no_show_risk_flag

# ── Strategies ───────────────────────────────────────────────────────

valid_gender = st.sampled_from(["M", "F"])
valid_age = st.integers(min_value=0, max_value=120)
invalid_age = st.integers(max_value=-1)

valid_binary = st.sampled_from([0, 1])
valid_handicap = st.integers(min_value=0, max_value=4)
valid_lead_time = st.integers(min_value=0, max_value=365)
valid_risk_score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
valid_category = st.sampled_from(["Low Risk", "Medium Risk", "High Risk"])
valid_neighbourhood = st.text(min_size=1, max_size=30)


def patient_strategy() -> st.SearchStrategy[Patient]:
    return st.builds(
        Patient,
        patient_id=st.text(min_size=1, max_size=10),
        age=valid_age,
        gender=valid_gender,
    )


def appointment_strategy() -> st.SearchStrategy[Appointment]:
    return st.builds(
        Appointment,
        appointment_id=st.text(min_size=1, max_size=10),
        patient=patient_strategy(),
        scheduled_day=st.just(datetime(2016, 4, 29)),
        appointment_day=st.just(datetime(2016, 5, 5)),
        neighbourhood=valid_neighbourhood,
        scholarship=valid_binary,
        hypertension=valid_binary,
        diabetes=valid_binary,
        alcoholism=valid_binary,
        handicap=valid_handicap,
        sms_received=valid_binary,
        lead_time_days=valid_lead_time,
    )


# ── Patient Properties ──────────────────────────────────────────────


class TestPatientProperties:
    @given(patient_id=st.text(min_size=1), age=valid_age, gender=valid_gender)
    @settings(max_examples=50)
    def test_valid_patient_always_constructs(
        self, patient_id: str, age: int, gender: str
    ) -> None:
        p = Patient(patient_id=patient_id, age=age, gender=gender)
        assert p.age >= 0
        assert p.gender in {"M", "F"}

    @given(age=invalid_age)
    @settings(max_examples=20)
    def test_negative_age_always_rejected(self, age: int) -> None:
        import pytest

        with pytest.raises(InvalidAppointmentDataError):
            Patient(patient_id="P1", age=age, gender="F")

    @given(gender=st.text().filter(lambda g: g not in {"M", "F"}))
    @settings(max_examples=20)
    def test_invalid_gender_always_rejected(self, gender: str) -> None:
        import pytest

        with pytest.raises(InvalidAppointmentDataError):
            Patient(patient_id="P1", age=25, gender=gender)


# ── Appointment Properties ───────────────────────────────────────────


class TestAppointmentProperties:
    @given(appt=appointment_strategy())
    @settings(max_examples=50)
    def test_valid_appointment_always_constructs(self, appt: Appointment) -> None:
        assert appt.lead_time_days >= 0
        assert appt.scholarship in {0, 1}
        assert appt.hypertension in {0, 1}
        assert appt.diabetes in {0, 1}
        assert appt.alcoholism in {0, 1}
        assert 0 <= appt.handicap <= 4
        assert appt.sms_received in {0, 1}

    @given(lead_time=st.integers(max_value=-1))
    @settings(max_examples=20)
    def test_negative_lead_time_always_rejected(self, lead_time: int) -> None:
        import pytest

        with pytest.raises(InvalidAppointmentDataError):
            Appointment(
                appointment_id="A1",
                patient=Patient(patient_id="P1", age=30, gender="F"),
                scheduled_day=datetime(2016, 4, 29),
                appointment_day=datetime(2016, 5, 5),
                neighbourhood="CENTRO",
                scholarship=0,
                hypertension=0,
                diabetes=0,
                alcoholism=0,
                handicap=0,
                sms_received=0,
                lead_time_days=lead_time,
            )


# ── NoShowOutcome Properties ────────────────────────────────────────


class TestNoShowOutcomeProperties:
    @given(score=valid_risk_score, category=valid_category)
    @settings(max_examples=50)
    def test_valid_outcome_always_constructs(
        self, score: float, category: str
    ) -> None:
        outcome = NoShowOutcome(
            appointment_id="A1",
            risk_score=score,
            risk_category=category,
            assessment_timestamp=datetime.now(),
            model_version="v1.0",
        )
        assert 0.0 <= outcome.risk_score <= 1.0

    @given(
        score=st.floats().filter(lambda x: x < 0.0 or x > 1.0).filter(
            lambda x: not (x != x)  # exclude NaN
        )
    )
    @settings(max_examples=20)
    def test_invalid_score_always_rejected(self, score: float) -> None:
        import pytest

        with pytest.raises(InvalidNoShowPredictionError):
            NoShowOutcome(
                appointment_id="A1",
                risk_score=score,
                risk_category="High Risk",
                assessment_timestamp=datetime.now(),
                model_version="v1.0",
            )


# ── Service Properties ──────────────────────────────────────────────


class TestBaselineServiceProperties:
    @given(appt=appointment_strategy())
    @settings(max_examples=100)
    def test_baseline_always_returns_valid_category(self, appt: Appointment) -> None:
        result = baseline_no_show_risk_flag(appt)
        assert result in {"Low Risk", "Medium Risk", "High Risk"}
