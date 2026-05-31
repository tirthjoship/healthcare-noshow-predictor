"""Tests for LogisticPredictor — train, predict, and error handling."""

from datetime import datetime

import pytest

from adapters.ml.calibrated_predictor import CalibratedPredictor
from adapters.ml.logistic_predictor import LogisticPredictor
from adapters.ml.xgboost_predictor import XGBoostPredictor
from domain.models import Appointment, NoShowOutcome, Patient

_SCHEDULED = datetime(2016, 1, 1, 8, 0, 0)
_APPOINTMENT = datetime(2016, 1, 10, 8, 0, 0)

_VALID_CATEGORIES = {"Low Risk", "Medium Risk", "High Risk"}


def _make_appointment(
    neighbourhood: str = "ALPHA",
    age: int = 40,
    gender: str = "F",
    lead_time_days: int = 9,
    scholarship: int = 0,
    hypertension: int = 0,
    diabetes: int = 0,
    alcoholism: int = 0,
    handicap: int = 0,
    sms_received: int = 0,
    appointment_id: str = "A1",
    patient_id: str = "P1",
) -> Appointment:
    patient = Patient(patient_id=patient_id, age=age, gender=gender)
    return Appointment(
        appointment_id=appointment_id,
        patient=patient,
        scheduled_day=_SCHEDULED,
        appointment_day=_APPOINTMENT,
        neighbourhood=neighbourhood,
        scholarship=scholarship,
        hypertension=hypertension,
        diabetes=diabetes,
        alcoholism=alcoholism,
        handicap=handicap,
        sms_received=sms_received,
        lead_time_days=lead_time_days,
    )


def _training_data() -> tuple[list[Appointment], list[bool]]:
    """Generate 30 appointments with ~33% no-show rate (10 no-shows)."""
    appointments: list[Appointment] = []
    labels: list[bool] = []

    neighbourhoods = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]

    for i in range(30):
        no_show = i % 3 == 0  # every 3rd = no-show → 10/30 = 33%
        appt = _make_appointment(
            appointment_id=f"APPT-{i}",
            patient_id=f"PAT-{i}",
            age=20 + (i % 60),
            gender="F" if i % 2 == 0 else "M",
            neighbourhood=neighbourhoods[i % len(neighbourhoods)],
            lead_time_days=i % 30,
            scholarship=i % 2,
            hypertension=1 if i % 4 == 0 else 0,
            diabetes=1 if i % 5 == 0 else 0,
            sms_received=1 if no_show else 0,  # reflect real-world confound
        )
        appointments.append(appt)
        labels.append(no_show)

    return appointments, labels


class TestLogisticPredictor:
    def test_train_and_predict(self) -> None:
        """Train on fixture data; predict returns a NoShowOutcome."""
        appointments, labels = _training_data()
        predictor = LogisticPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="TEST-1", patient_id="PTEST-1")
        result = predictor.predict_no_show(test_appt)

        assert isinstance(result, NoShowOutcome)
        assert result.appointment_id == "TEST-1"

    def test_prediction_score_in_range(self) -> None:
        """Predicted risk score must be in [0.0, 1.0]."""
        appointments, labels = _training_data()
        predictor = LogisticPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="TEST-2", patient_id="PTEST-2")
        result = predictor.predict_no_show(test_appt)

        assert 0.0 <= result.risk_score <= 1.0

    def test_prediction_has_valid_category(self) -> None:
        """Risk category must be one of the three valid strings."""
        appointments, labels = _training_data()
        predictor = LogisticPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="TEST-3", patient_id="PTEST-3")
        result = predictor.predict_no_show(test_appt)

        assert result.risk_category in _VALID_CATEGORIES

    def test_prediction_model_version(self) -> None:
        """Model version must be 'logistic-v1'."""
        appointments, labels = _training_data()
        predictor = LogisticPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="TEST-4", patient_id="PTEST-4")
        result = predictor.predict_no_show(test_appt)

        assert result.model_version == "logistic-v1"

    def test_predict_without_train_raises(self) -> None:
        """Calling predict_no_show before train must raise RuntimeError."""
        predictor = LogisticPredictor()
        test_appt = _make_appointment(appointment_id="TEST-5", patient_id="PTEST-5")

        with pytest.raises(RuntimeError):
            predictor.predict_no_show(test_appt)


class TestXGBoostPredictor:
    def test_train_and_predict(self) -> None:
        """Train on fixture data; predict returns a NoShowOutcome."""
        appointments, labels = _training_data()
        predictor = XGBoostPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="XGB-1", patient_id="PXGB-1")
        result = predictor.predict_no_show(test_appt)

        assert isinstance(result, NoShowOutcome)
        assert result.appointment_id == "XGB-1"

    def test_prediction_score_in_range(self) -> None:
        """Predicted risk score must be in [0.0, 1.0]."""
        appointments, labels = _training_data()
        predictor = XGBoostPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="XGB-2", patient_id="PXGB-2")
        result = predictor.predict_no_show(test_appt)

        assert 0.0 <= result.risk_score <= 1.0

    def test_prediction_has_valid_category(self) -> None:
        """Risk category must be one of the three valid strings."""
        appointments, labels = _training_data()
        predictor = XGBoostPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="XGB-3", patient_id="PXGB-3")
        result = predictor.predict_no_show(test_appt)

        assert result.risk_category in _VALID_CATEGORIES

    def test_prediction_model_version(self) -> None:
        """Model version must be 'xgboost-v1'."""
        appointments, labels = _training_data()
        predictor = XGBoostPredictor()
        predictor.train(appointments, labels)

        test_appt = _make_appointment(appointment_id="XGB-4", patient_id="PXGB-4")
        result = predictor.predict_no_show(test_appt)

        assert result.model_version == "xgboost-v1"

    def test_predict_without_train_raises(self) -> None:
        """Calling predict_no_show before train must raise RuntimeError."""
        predictor = XGBoostPredictor()
        test_appt = _make_appointment(appointment_id="XGB-5", patient_id="PXGB-5")

        with pytest.raises(RuntimeError):
            predictor.predict_no_show(test_appt)


class TestCalibratedPredictor:
    def _large_training_data(self) -> tuple[list[Appointment], list[bool]]:
        """Generate 60 appointments for CalibratedClassifierCV (needs more data for CV)."""
        appts: list[Appointment] = []
        labels: list[bool] = []
        for i in range(60):
            appt = _make_appointment(
                age=20 + (i % 50),
                lead_time_days=i,
                sms_received=i % 2,
                neighbourhood="A" if i < 30 else "B",
                patient_id=f"P{i}",
                appointment_id=f"A{i}",
            )
            appts.append(appt)
            labels.append(i % 3 == 0)
        return appts, labels

    def test_train_and_predict(self) -> None:
        """Train on 60-row fixture; predict returns a NoShowOutcome."""
        appts, labels = self._large_training_data()
        predictor = CalibratedPredictor()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert isinstance(result, NoShowOutcome)

    def test_prediction_score_in_range(self) -> None:
        """Calibrated risk score must be in [0.0, 1.0]."""
        appts, labels = self._large_training_data()
        predictor = CalibratedPredictor()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert 0.0 <= result.risk_score <= 1.0

    def test_model_version(self) -> None:
        """Model version must be 'calibrated-xgboost-v1'."""
        appts, labels = self._large_training_data()
        predictor = CalibratedPredictor()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert result.model_version == "calibrated-xgboost-v1"
