"""Tests for FeatureEncoder — target encoding, shape, and feature values."""

from datetime import datetime

import numpy as np
import pytest

from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Appointment, Patient

_SCHEDULED = datetime(2016, 1, 1, 8, 0, 0)
_APPOINTMENT = datetime(2016, 1, 10, 8, 0, 0)


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


class TestFeatureEncoderShape:
    def test_produces_10_columns(self) -> None:
        enc = FeatureEncoder()
        appts = [_make_appointment()]
        labels = [True]
        X = enc.fit_transform(appts, labels)
        assert X.shape == (1, 10)

    def test_feature_names_length(self) -> None:
        enc = FeatureEncoder()
        assert len(enc.get_feature_names()) == 10

    def test_feature_names_content(self) -> None:
        enc = FeatureEncoder()
        expected = [
            "age",
            "gender_enc",
            "lead_time_days",
            "scholarship",
            "hypertension",
            "diabetes",
            "alcoholism",
            "handicap",
            "sms_received",
            "neighbourhood_enc",
        ]
        assert enc.get_feature_names() == expected


class TestTargetEncoding:
    def test_neighbourhood_encoded_from_training_labels(self) -> None:
        # Neighbourhood A: all no-show → 1.0, Neighbourhood B: none → 0.0
        appts_train = [
            _make_appointment(neighbourhood="A", appointment_id="1", patient_id="P1"),
            _make_appointment(neighbourhood="A", appointment_id="2", patient_id="P2"),
            _make_appointment(neighbourhood="B", appointment_id="3", patient_id="P3"),
            _make_appointment(neighbourhood="B", appointment_id="4", patient_id="P4"),
        ]
        labels = [True, True, False, False]
        enc = FeatureEncoder()
        enc.fit(appts_train, labels)

        appt_a = _make_appointment(
            neighbourhood="A", appointment_id="5", patient_id="P5"
        )
        appt_b = _make_appointment(
            neighbourhood="B", appointment_id="6", patient_id="P6"
        )

        X_a = enc.transform([appt_a])
        X_b = enc.transform([appt_b])

        assert X_a[0, 9] == pytest.approx(1.0)
        assert X_b[0, 9] == pytest.approx(0.0)

    def test_unseen_neighbourhood_gets_global_mean(self) -> None:
        appts_train = [
            _make_appointment(neighbourhood="A", appointment_id="1", patient_id="P1"),
            _make_appointment(neighbourhood="A", appointment_id="2", patient_id="P2"),
            _make_appointment(neighbourhood="B", appointment_id="3", patient_id="P3"),
        ]
        labels = [True, True, False]
        enc = FeatureEncoder()
        enc.fit(appts_train, labels)

        # global mean = 2/3
        appt_unseen = _make_appointment(
            neighbourhood="UNSEEN", appointment_id="7", patient_id="P7"
        )
        X = enc.transform([appt_unseen])
        assert X[0, 9] == pytest.approx(2 / 3)

    def test_transform_without_fit_raises(self) -> None:
        enc = FeatureEncoder()
        appt = _make_appointment()
        with pytest.raises(RuntimeError):
            enc.transform([appt])


class TestFeatureValues:
    def test_gender_encoding(self) -> None:
        enc = FeatureEncoder()
        female = _make_appointment(gender="F", appointment_id="F1", patient_id="PF")
        male = _make_appointment(gender="M", appointment_id="M1", patient_id="PM")
        enc.fit([female, male], [False, False])

        X_f = enc.transform([female])
        X_m = enc.transform([male])

        assert X_f[0, 1] == pytest.approx(1.0)
        assert X_m[0, 1] == pytest.approx(0.0)

    def test_numeric_features_passthrough(self) -> None:
        enc = FeatureEncoder()
        appt = _make_appointment(
            age=55,
            lead_time_days=14,
            sms_received=1,
            handicap=2,
            appointment_id="N1",
            patient_id="PN",
        )
        enc.fit([appt], [True])
        X = enc.transform([appt])

        assert X[0, 0] == pytest.approx(55.0)  # age
        assert X[0, 2] == pytest.approx(14.0)  # lead_time_days
        assert X[0, 8] == pytest.approx(1.0)  # sms_received
        assert X[0, 7] == pytest.approx(2.0)  # handicap
