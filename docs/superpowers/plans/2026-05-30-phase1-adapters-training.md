# Phase 1: Adapters + Model Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build CSV adapter, feature encoder, 3 predictors, evaluation metrics, and training pipeline. Output: `reports/model_metrics.json` with GroupKFold + temporal holdout results.

**Architecture:** Hexagonal — adapters implement `AppointmentRepository` and `NoShowPredictorPort` protocols from `domain/ports.py`. Application layer orchestrates training pipeline. Domain layer untouched.

**Tech Stack:** Python 3.12, pandas, scikit-learn, XGBoost, numpy, pytest, Hypothesis

**Spec:** `docs/superpowers/specs/2026-05-30-phase1-adapters-training-design.md`
**ADRs:** `docs/adr/ADR-007` (target encoding), `ADR-008` (model lineup), `ADR-011` (split strategy)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `adapters/data/csv_repository.py` | Load Kaggle CSV → domain objects, leakage guard |
| `adapters/ml/feature_encoder.py` | Appointment → 10-feature numpy array, target encoding |
| `adapters/ml/logistic_predictor.py` | LogisticRegression wrapper implementing NoShowPredictorPort |
| `adapters/ml/xgboost_predictor.py` | XGBClassifier wrapper implementing NoShowPredictorPort |
| `adapters/ml/calibrated_predictor.py` | Isotonic calibration wrapper around any predictor |
| `adapters/ml/evaluation.py` | Metric computation: AUC, F1, Brier, precision@K, ECE |
| `application/use_cases.py` | Updated: `train_and_evaluate()` with GroupKFold + temporal |
| `tests/test_csv_repository.py` | CSV loading, leakage, date filtering |
| `tests/test_feature_encoder.py` | Encoding correctness, target encoding isolation |
| `tests/test_predictors.py` | All 3 predictors train/predict on fixture |
| `tests/test_evaluation.py` | Metric computation on known arrays |

---

## Task 1: KaggleAppointmentCSVRepository

**Files:**
- Create: `adapters/data/csv_repository.py`
- Create: `tests/test_csv_repository.py`
- Read: `domain/ports.py` (AppointmentRepository protocol)
- Read: `domain/models.py` (Patient, Appointment)

**Parallelizable with:** Task 2 (no dependencies)

- [ ] **Step 1: Write test fixtures — small CSV helper**

Create `tests/conftest.py` with a shared fixture that writes a small CSV to `tmp_path`:

```python
# tests/conftest.py
"""Shared test fixtures for healthcare-noshow-predictor."""

from pathlib import Path

import pytest

SAMPLE_CSV_CONTENT = """\
PatientId,AppointmentID,Gender,ScheduledDay,AppointmentDay,Age,Neighbourhood,Scholarship,Hipertension,Diabetes,Alcoholism,Handcap,SMS_received,No-show
1.0,1001,F,2016-04-29T18:38:08Z,2016-04-29T00:00:00Z,62,JARDIM DA PENHA,0,1,0,0,0,0,No
1.0,1002,F,2016-04-30T10:00:00Z,2016-05-05T00:00:00Z,62,JARDIM DA PENHA,0,1,0,0,0,1,Yes
2.0,1003,M,2016-04-29T08:00:00Z,2016-05-10T00:00:00Z,25,CENTRO,1,0,0,0,0,0,No
3.0,1004,F,2016-05-01T12:00:00Z,2016-05-15T00:00:00Z,35,RESISTÊNCIA,0,0,1,0,1,1,Yes
4.0,1005,M,2016-05-02T09:00:00Z,2016-06-01T00:00:00Z,45,CENTRO,0,0,0,1,0,0,No
4.0,1006,M,2016-05-03T14:00:00Z,2016-06-05T00:00:00Z,45,CENTRO,0,0,0,1,0,1,Yes
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a small CSV fixture and return the path."""
    csv_path = tmp_path / "KaggleV2-May-2016.csv"
    csv_path.write_text(SAMPLE_CSV_CONTENT)
    return csv_path
```

- [ ] **Step 2: Write failing tests for CSV repository**

```python
# tests/test_csv_repository.py
"""Tests for KaggleAppointmentCSVRepository."""

from datetime import datetime
from pathlib import Path

import pytest

from adapters.data.csv_repository import KaggleAppointmentCSVRepository
from domain.exceptions import DataLeakageError
from domain.models import Appointment, Patient


class TestCSVRepositoryLoading:
    def test_loads_all_rows(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        assert len(appointments) == 6

    def test_returns_appointment_objects(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appt = repo.get_appointments()[0]
        assert isinstance(appt, Appointment)
        assert isinstance(appt.patient, Patient)

    def test_patient_fields_parsed(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appt = repo.get_appointments()[0]
        assert appt.patient.patient_id == "1"
        assert appt.patient.age == 62
        assert appt.patient.gender == "F"

    def test_appointment_fields_parsed(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appt = repo.get_appointments()[0]
        assert appt.appointment_id == "1001"
        assert appt.neighbourhood == "JARDIM DA PENHA"
        assert appt.hypertension == 1
        assert appt.sms_received == 0

    def test_lead_time_computed(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appt = repo.get_appointments()[1]
        # Scheduled Apr 30, Appointment May 5 = 5 days
        assert appt.lead_time_days == 5

    def test_negative_lead_time_clipped_to_zero(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        # First row: same-day, ScheduledDay has timestamp after midnight
        appt = repo.get_appointments()[0]
        assert appt.lead_time_days >= 0


class TestCSVRepositoryLabels:
    def test_labels_aligned_with_appointments(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        labels = repo.get_labels()
        assert len(labels) == len(appointments)

    def test_labels_are_booleans(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        labels = repo.get_labels()
        assert all(isinstance(label, bool) for label in labels)

    def test_noshow_yes_maps_to_true(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        labels = repo.get_labels()
        # Row index 1 is "Yes" (no-show)
        assert labels[1] is True

    def test_noshow_no_maps_to_false(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        labels = repo.get_labels()
        # Row index 0 is "No" (showed up)
        assert labels[0] is False


class TestCSVRepositoryDateFiltering:
    def test_filter_by_start_date(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments(
            start_date=datetime(2016, 6, 1)
        )
        # Only June appointments (rows 4 and 5)
        assert len(appointments) == 2

    def test_filter_by_end_date(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments(
            end_date=datetime(2016, 5, 1)
        )
        # Only April 29 appointment (row 0)
        assert len(appointments) == 1

    def test_labels_filtered_same_as_appointments(self, sample_csv: Path) -> None:
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments(start_date=datetime(2016, 6, 1))
        labels = repo.get_labels(start_date=datetime(2016, 6, 1))
        assert len(labels) == len(appointments)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_csv_repository.py -v`
Expected: ImportError — `adapters.data.csv_repository` does not exist yet.

- [ ] **Step 4: Implement KaggleAppointmentCSVRepository**

```python
# adapters/data/csv_repository.py
"""CSV repository adapter for Kaggle Medical Appointments dataset."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from domain.exceptions import DataLeakageError
from domain.models import Appointment, Patient

# Columns that would constitute data leakage if used as features.
# The "No-show" column is the target — used only for labels.
LABEL_COLUMN = "No-show"
ALLOWED_COLUMNS = frozenset(
    {
        "PatientId",
        "AppointmentID",
        "Gender",
        "ScheduledDay",
        "AppointmentDay",
        "Age",
        "Neighbourhood",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "No-show",
    }
)


class KaggleAppointmentCSVRepository:
    """Load Kaggle Medical Appointments CSV into domain objects.

    Implements AppointmentRepository protocol from domain/ports.py.
    All features are knowable at scheduling time — no post-appointment data.
    """

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._df = self._load_and_validate()

    def _load_and_validate(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv_path)

        # Leakage guard: reject unexpected columns
        extra_cols = set(df.columns) - ALLOWED_COLUMNS
        if extra_cols:
            raise DataLeakageError(
                f"Unexpected columns detected (potential leakage): {extra_cols}"
            )

        # Parse dates
        df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

        # Compute lead time, clip negatives to 0
        df["lead_time_days"] = (
            df["AppointmentDay"] - df["ScheduledDay"]
        ).dt.days.clip(lower=0)

        # Map target
        df["no_show"] = df[LABEL_COLUMN] == "Yes"

        return df

    def _filter(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        df = self._df
        if start_date is not None:
            df = df[df["AppointmentDay"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["AppointmentDay"] < pd.Timestamp(end_date)]
        return df

    def get_appointments(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Appointment]:
        df = self._filter(start_date, end_date)
        appointments: list[Appointment] = []
        for _, row in df.iterrows():
            patient = Patient(
                patient_id=str(int(row["PatientId"])),
                age=int(row["Age"]),
                gender=str(row["Gender"]),
            )
            appt = Appointment(
                appointment_id=str(int(row["AppointmentID"])),
                patient=patient,
                scheduled_day=row["ScheduledDay"].to_pydatetime(),
                appointment_day=row["AppointmentDay"].to_pydatetime(),
                neighbourhood=str(row["Neighbourhood"]),
                scholarship=int(row["Scholarship"]),
                hypertension=int(row["Hipertension"]),
                diabetes=int(row["Diabetes"]),
                alcoholism=int(row["Alcoholism"]),
                handicap=int(row["Handcap"]),
                sms_received=int(row["SMS_received"]),
                lead_time_days=int(row["lead_time_days"]),
            )
            appointments.append(appt)
        return appointments

    def get_labels(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[bool]:
        df = self._filter(start_date, end_date)
        return [bool(v) for v in df["no_show"].tolist()]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_csv_repository.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_csv_repository.py adapters/data/csv_repository.py
git commit -m "feat: KaggleAppointmentCSVRepository — CSV loading with leakage guard"
```

---

## Task 2: FeatureEncoder

**Files:**
- Create: `adapters/ml/feature_encoder.py`
- Create: `tests/test_feature_encoder.py`

**Parallelizable with:** Task 1 (no dependencies)

- [ ] **Step 1: Write failing tests for FeatureEncoder**

```python
# tests/test_feature_encoder.py
"""Tests for FeatureEncoder with target encoding."""

from datetime import datetime

import numpy as np
import pytest

from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Appointment, Patient


def _make_appointment(
    neighbourhood: str = "CENTRO",
    age: int = 30,
    gender: str = "F",
    lead_time: int = 5,
    sms: int = 0,
    scholarship: int = 0,
    hypertension: int = 0,
    diabetes: int = 0,
    alcoholism: int = 0,
    handicap: int = 0,
    patient_id: str = "P1",
    appt_id: str = "A1",
) -> Appointment:
    return Appointment(
        appointment_id=appt_id,
        patient=Patient(patient_id=patient_id, age=age, gender=gender),
        scheduled_day=datetime(2016, 4, 29),
        appointment_day=datetime(2016, 5, 4),
        neighbourhood=neighbourhood,
        scholarship=scholarship,
        hypertension=hypertension,
        diabetes=diabetes,
        alcoholism=alcoholism,
        handicap=handicap,
        sms_received=sms,
        lead_time_days=lead_time,
    )


class TestFeatureEncoderShape:
    def test_produces_10_columns(self) -> None:
        encoder = FeatureEncoder()
        appts = [_make_appointment() for _ in range(5)]
        labels = [True, False, True, False, True]
        X = encoder.fit_transform(appts, labels)
        assert X.shape == (5, 10)

    def test_feature_names_length(self) -> None:
        encoder = FeatureEncoder()
        appts = [_make_appointment() for _ in range(3)]
        labels = [True, False, True]
        encoder.fit(appts, labels)
        assert len(encoder.get_feature_names()) == 10

    def test_feature_names_content(self) -> None:
        encoder = FeatureEncoder()
        appts = [_make_appointment() for _ in range(3)]
        labels = [True, False, True]
        encoder.fit(appts, labels)
        names = encoder.get_feature_names()
        assert "age" in names
        assert "lead_time_days" in names
        assert "neighbourhood_enc" in names


class TestTargetEncoding:
    def test_neighbourhood_encoded_from_training_labels(self) -> None:
        appts = [
            _make_appointment(neighbourhood="A"),
            _make_appointment(neighbourhood="A"),
            _make_appointment(neighbourhood="B"),
            _make_appointment(neighbourhood="B"),
        ]
        labels = [True, True, False, False]  # A=1.0, B=0.0
        encoder = FeatureEncoder()
        X = encoder.fit_transform(appts, labels)
        neigh_idx = encoder.get_feature_names().index("neighbourhood_enc")
        # A should encode to 1.0 (all no-show), B to 0.0 (none)
        assert X[0, neigh_idx] == pytest.approx(1.0)
        assert X[2, neigh_idx] == pytest.approx(0.0)

    def test_unseen_neighbourhood_gets_global_mean(self) -> None:
        train_appts = [
            _make_appointment(neighbourhood="A"),
            _make_appointment(neighbourhood="A"),
        ]
        train_labels = [True, False]  # global mean = 0.5
        test_appts = [_make_appointment(neighbourhood="UNKNOWN")]
        encoder = FeatureEncoder()
        encoder.fit(train_appts, train_labels)
        X_test = encoder.transform(test_appts)
        neigh_idx = encoder.get_feature_names().index("neighbourhood_enc")
        assert X_test[0, neigh_idx] == pytest.approx(0.5)

    def test_transform_without_fit_raises(self) -> None:
        encoder = FeatureEncoder()
        with pytest.raises(RuntimeError, match="fit"):
            encoder.transform([_make_appointment()])


class TestFeatureValues:
    def test_gender_encoding(self) -> None:
        encoder = FeatureEncoder()
        appts = [
            _make_appointment(gender="F"),
            _make_appointment(gender="M"),
        ]
        labels = [True, False]
        X = encoder.fit_transform(appts, labels)
        gender_idx = encoder.get_feature_names().index("gender_enc")
        assert X[0, gender_idx] == 1.0  # F -> 1
        assert X[1, gender_idx] == 0.0  # M -> 0

    def test_numeric_features_passthrough(self) -> None:
        encoder = FeatureEncoder()
        appt = _make_appointment(age=45, lead_time=10, sms=1, handicap=2)
        X = encoder.fit_transform([appt], [True])
        names = encoder.get_feature_names()
        assert X[0, names.index("age")] == 45
        assert X[0, names.index("lead_time_days")] == 10
        assert X[0, names.index("sms_received")] == 1
        assert X[0, names.index("handicap")] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_feature_encoder.py -v`
Expected: ImportError — `adapters.ml.feature_encoder` does not exist yet.

- [ ] **Step 3: Implement FeatureEncoder**

```python
# adapters/ml/feature_encoder.py
"""Feature encoder: Appointment domain objects → numeric feature matrix.

Target-encodes neighbourhood using training labels only (ADR-007).
"""

import numpy as np

from domain.models import Appointment


class FeatureEncoder:
    """Transform Appointment objects into a 10-feature numpy array.

    Features: age, gender_enc, lead_time_days, scholarship, hypertension,
    diabetes, alcoholism, handicap, sms_received, neighbourhood_enc.

    Target encoding for neighbourhood is fit on training data only.
    Unseen neighbourhoods at transform time get the global mean.
    """

    FEATURE_NAMES: list[str] = [
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

    def __init__(self) -> None:
        self._neighbourhood_map: dict[str, float] = {}
        self._global_mean: float = 0.0
        self._is_fitted: bool = False

    def fit(
        self, appointments: list[Appointment], labels: list[bool]
    ) -> None:
        """Fit target encoding from training data.

        Args:
            appointments: Training appointments.
            labels: Aligned boolean labels (True = no-show).
        """
        from collections import defaultdict

        neigh_sums: dict[str, float] = defaultdict(float)
        neigh_counts: dict[str, int] = defaultdict(int)

        for appt, label in zip(appointments, labels):
            neigh_sums[appt.neighbourhood] += float(label)
            neigh_counts[appt.neighbourhood] += 1

        self._neighbourhood_map = {
            n: neigh_sums[n] / neigh_counts[n] for n in neigh_sums
        }
        total_positive = sum(float(lb) for lb in labels)
        self._global_mean = total_positive / len(labels) if labels else 0.0
        self._is_fitted = True

    def transform(self, appointments: list[Appointment]) -> np.ndarray:
        """Transform appointments into feature matrix.

        Args:
            appointments: Appointments to encode.

        Returns:
            numpy array of shape (n_appointments, 10).

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEncoder must be fit before transform. Call fit() first."
            )

        rows: list[list[float]] = []
        for appt in appointments:
            neigh_enc = self._neighbourhood_map.get(
                appt.neighbourhood, self._global_mean
            )
            row = [
                float(appt.patient.age),
                1.0 if appt.patient.gender == "F" else 0.0,
                float(appt.lead_time_days),
                float(appt.scholarship),
                float(appt.hypertension),
                float(appt.diabetes),
                float(appt.alcoholism),
                float(appt.handicap),
                float(appt.sms_received),
                neigh_enc,
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float64)

    def fit_transform(
        self, appointments: list[Appointment], labels: list[bool]
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(appointments, labels)
        return self.transform(appointments)

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        return list(self.FEATURE_NAMES)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_feature_encoder.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/feature_encoder.py tests/test_feature_encoder.py
git commit -m "feat: FeatureEncoder with train-only target encoding for neighbourhood"
```

---

## Task 3: Evaluation Metrics

**Files:**
- Create: `adapters/ml/evaluation.py`
- Create: `tests/test_evaluation.py`

**Parallelizable with:** Tasks 1 and 2 (no dependencies)

- [ ] **Step 1: Write failing tests for evaluation**

```python
# tests/test_evaluation.py
"""Tests for model evaluation metrics."""

import numpy as np
import pytest

from adapters.ml.evaluation import evaluate_model


class TestEvaluateModel:
    def test_returns_required_keys(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        result = evaluate_model(y_true, y_prob, "test_model")
        assert "auc" in result
        assert "f1" in result
        assert "brier" in result
        assert "precision_at_k" in result
        assert "ece" in result

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        result = evaluate_model(y_true, y_prob, "perfect", k=3)
        assert result["auc"] == pytest.approx(1.0)
        assert result["brier"] == pytest.approx(0.0)
        assert result["precision_at_k"] == pytest.approx(1.0)

    def test_auc_bounded(self) -> None:
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6, 0.8, 0.2])
        result = evaluate_model(y_true, y_prob, "bounded")
        assert 0.0 <= result["auc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["brier"] <= 1.0
        assert 0.0 <= result["ece"] <= 1.0

    def test_precision_at_k(self) -> None:
        y_true = np.array([1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
        # Top 2 by prob: indices 0 (true=1) and 3 (true=1) => precision = 1.0
        result = evaluate_model(y_true, y_prob, "topk", k=2)
        assert result["precision_at_k"] == pytest.approx(1.0)

    def test_model_name_in_result(self) -> None:
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        result = evaluate_model(y_true, y_prob, "my_model")
        assert result["model_name"] == "my_model"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement evaluate_model**

```python
# adapters/ml/evaluation.py
"""Model evaluation metrics for no-show prediction."""

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    k: int = 20,
) -> dict[str, float | str]:
    """Compute evaluation metrics for a no-show prediction model.

    Args:
        y_true: Binary ground truth (0 = showed, 1 = no-show).
        y_prob: Predicted probability of no-show.
        model_name: Identifier for the model.
        k: Top-K for precision@K (daily call list size).

    Returns:
        Dict with auc, f1, brier, precision_at_k, ece, model_name.
    """
    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # F1 at optimal threshold (maximize F1 over thresholds)
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
    f1 = best_f1

    # Precision@K
    top_k_indices = np.argsort(y_prob)[-k:]
    precision_at_k = float(y_true[top_k_indices].mean()) if k > 0 else 0.0

    # Expected Calibration Error (10 bins)
    ece = _expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "model_name": model_name,
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "brier": round(brier, 4),
        "precision_at_k": round(precision_at_k, 4),
        "ece": round(ece, 4),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return ece
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/evaluation.py tests/test_evaluation.py
git commit -m "feat: evaluate_model — AUC, F1, Brier, precision@K, ECE metrics"
```

---

## Task 4: LogisticPredictor

**Files:**
- Create: `adapters/ml/logistic_predictor.py`
- Create: `tests/test_predictors.py`

**Depends on:** Task 2 (FeatureEncoder)

- [ ] **Step 1: Write failing tests for LogisticPredictor**

```python
# tests/test_predictors.py
"""Tests for ML predictor adapters (Logistic, XGBoost, Calibrated)."""

from datetime import datetime

import pytest

from adapters.ml.logistic_predictor import LogisticPredictor
from domain.models import Appointment, NoShowOutcome, Patient


def _make_appointment(
    neighbourhood: str = "CENTRO",
    age: int = 30,
    gender: str = "F",
    lead_time: int = 5,
    sms: int = 0,
    scholarship: int = 0,
    hypertension: int = 0,
    diabetes: int = 0,
    alcoholism: int = 0,
    handicap: int = 0,
    patient_id: str = "P1",
    appt_id: str = "A1",
) -> Appointment:
    return Appointment(
        appointment_id=appt_id,
        patient=Patient(patient_id=patient_id, age=age, gender=gender),
        scheduled_day=datetime(2016, 4, 29),
        appointment_day=datetime(2016, 5, 4),
        neighbourhood=neighbourhood,
        scholarship=scholarship,
        hypertension=hypertension,
        diabetes=diabetes,
        alcoholism=alcoholism,
        handicap=handicap,
        sms_received=sms,
        lead_time_days=lead_time,
    )


def _training_data() -> tuple[list[Appointment], list[bool]]:
    """Generate 30-row training fixture with mix of no-show/show."""
    appointments = []
    labels = []
    for i in range(30):
        is_noshow = i % 3 == 0  # ~33% no-show rate
        appt = _make_appointment(
            age=20 + i,
            lead_time=i * 2,
            sms=1 if i % 2 == 0 else 0,
            neighbourhood="A" if i < 15 else "B",
            patient_id=f"P{i}",
            appt_id=f"A{i}",
        )
        appointments.append(appt)
        labels.append(is_noshow)
    return appointments, labels


class TestLogisticPredictor:
    def test_train_and_predict(self) -> None:
        predictor = LogisticPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert isinstance(result, NoShowOutcome)

    def test_prediction_score_in_range(self) -> None:
        predictor = LogisticPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert 0.0 <= result.risk_score <= 1.0

    def test_prediction_has_valid_category(self) -> None:
        predictor = LogisticPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert result.risk_category in {"Low Risk", "Medium Risk", "High Risk"}

    def test_prediction_model_version(self) -> None:
        predictor = LogisticPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert result.model_version == "logistic-v1"

    def test_predict_without_train_raises(self) -> None:
        predictor = LogisticPredictor()
        with pytest.raises(RuntimeError, match="train"):
            predictor.predict_no_show(_make_appointment())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predictors.py::TestLogisticPredictor -v`
Expected: ImportError.

- [ ] **Step 3: Implement LogisticPredictor**

```python
# adapters/ml/logistic_predictor.py
"""Logistic regression predictor for appointment no-show."""

from datetime import datetime

from sklearn.linear_model import LogisticRegression

from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Appointment, NoShowOutcome


def _score_to_category(score: float) -> str:
    """Map risk score to category."""
    if score < 0.2:
        return "Low Risk"
    if score <= 0.4:
        return "Medium Risk"
    return "High Risk"


class LogisticPredictor:
    """Logistic regression wrapper implementing NoShowPredictorPort.

    Uses FeatureEncoder internally for feature extraction.
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._encoder = FeatureEncoder()

    def train(
        self, appointments: list[Appointment], outcomes: list[bool]
    ) -> None:
        """Train logistic regression on appointments."""
        X = self._encoder.fit_transform(appointments, outcomes)
        y = [int(o) for o in outcomes]
        self._model = LogisticRegression(max_iter=1000, random_state=42)
        self._model.fit(X, y)

    def predict_no_show(self, appointment: Appointment) -> NoShowOutcome:
        """Predict no-show probability for one appointment."""
        if self._model is None:
            raise RuntimeError(
                "Model must be trained before prediction. Call train() first."
            )
        X = self._encoder.transform([appointment])
        prob = float(self._model.predict_proba(X)[0, 1])
        return NoShowOutcome(
            appointment_id=appointment.appointment_id,
            risk_score=prob,
            risk_category=_score_to_category(prob),
            assessment_timestamp=datetime.now(),
            model_version="logistic-v1",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predictors.py::TestLogisticPredictor -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adapters/ml/logistic_predictor.py tests/test_predictors.py
git commit -m "feat: LogisticPredictor implementing NoShowPredictorPort"
```

---

## Task 5: XGBoostPredictor

**Files:**
- Create: `adapters/ml/xgboost_predictor.py`
- Modify: `tests/test_predictors.py` (add TestXGBoostPredictor class)

**Depends on:** Task 2 (FeatureEncoder)

- [ ] **Step 1: Add XGBoost tests to test_predictors.py**

Append to `tests/test_predictors.py`:

```python
from adapters.ml.xgboost_predictor import XGBoostPredictor


class TestXGBoostPredictor:
    def test_train_and_predict(self) -> None:
        predictor = XGBoostPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert isinstance(result, NoShowOutcome)

    def test_prediction_score_in_range(self) -> None:
        predictor = XGBoostPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert 0.0 <= result.risk_score <= 1.0

    def test_prediction_has_valid_category(self) -> None:
        predictor = XGBoostPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert result.risk_category in {"Low Risk", "Medium Risk", "High Risk"}

    def test_model_version(self) -> None:
        predictor = XGBoostPredictor()
        appts, labels = _training_data()
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert result.model_version == "xgboost-v1"

    def test_predict_without_train_raises(self) -> None:
        predictor = XGBoostPredictor()
        with pytest.raises(RuntimeError, match="train"):
            predictor.predict_no_show(_make_appointment())
```

- [ ] **Step 2: Implement XGBoostPredictor**

```python
# adapters/ml/xgboost_predictor.py
"""XGBoost predictor for appointment no-show."""

from datetime import datetime

from xgboost import XGBClassifier

from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.logistic_predictor import _score_to_category
from domain.models import Appointment, NoShowOutcome


class XGBoostPredictor:
    """XGBoost wrapper implementing NoShowPredictorPort."""

    def __init__(self) -> None:
        self._model: XGBClassifier | None = None
        self._encoder = FeatureEncoder()

    def train(
        self, appointments: list[Appointment], outcomes: list[bool]
    ) -> None:
        """Train XGBoost on appointments."""
        X = self._encoder.fit_transform(appointments, outcomes)
        y = [int(o) for o in outcomes]
        self._model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            scale_pos_weight=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)

    def predict_no_show(self, appointment: Appointment) -> NoShowOutcome:
        """Predict no-show probability for one appointment."""
        if self._model is None:
            raise RuntimeError(
                "Model must be trained before prediction. Call train() first."
            )
        X = self._encoder.transform([appointment])
        prob = float(self._model.predict_proba(X)[0, 1])
        return NoShowOutcome(
            appointment_id=appointment.appointment_id,
            risk_score=prob,
            risk_category=_score_to_category(prob),
            assessment_timestamp=datetime.now(),
            model_version="xgboost-v1",
        )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_predictors.py -v`
Expected: All 10 tests PASS (5 Logistic + 5 XGBoost).

- [ ] **Step 4: Commit**

```bash
git add adapters/ml/xgboost_predictor.py tests/test_predictors.py
git commit -m "feat: XGBoostPredictor implementing NoShowPredictorPort"
```

---

## Task 6: CalibratedPredictor

**Files:**
- Create: `adapters/ml/calibrated_predictor.py`
- Modify: `tests/test_predictors.py` (add TestCalibratedPredictor class)

**Depends on:** Task 5 (XGBoostPredictor)

- [ ] **Step 1: Add Calibrated tests to test_predictors.py**

Append to `tests/test_predictors.py`:

```python
from adapters.ml.calibrated_predictor import CalibratedPredictor
from adapters.ml.xgboost_predictor import XGBoostPredictor as XGB


class TestCalibratedPredictor:
    def test_train_and_predict(self) -> None:
        # Need more data for calibration CV
        appts, labels = [], []
        for i in range(60):
            appt = _make_appointment(
                age=20 + (i % 50),
                lead_time=i,
                sms=i % 2,
                neighbourhood="A" if i < 30 else "B",
                patient_id=f"P{i}",
                appt_id=f"A{i}",
            )
            appts.append(appt)
            labels.append(i % 3 == 0)
        predictor = CalibratedPredictor(base_predictor_cls=XGB)
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert isinstance(result, NoShowOutcome)

    def test_prediction_score_in_range(self) -> None:
        appts, labels = [], []
        for i in range(60):
            appt = _make_appointment(
                age=20 + (i % 50),
                lead_time=i,
                sms=i % 2,
                neighbourhood="A" if i < 30 else "B",
                patient_id=f"P{i}",
                appt_id=f"A{i}",
            )
            appts.append(appt)
            labels.append(i % 3 == 0)
        predictor = CalibratedPredictor(base_predictor_cls=XGB)
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert 0.0 <= result.risk_score <= 1.0

    def test_model_version(self) -> None:
        appts, labels = [], []
        for i in range(60):
            appt = _make_appointment(
                age=20 + (i % 50),
                lead_time=i,
                sms=i % 2,
                neighbourhood="A" if i < 30 else "B",
                patient_id=f"P{i}",
                appt_id=f"A{i}",
            )
            appts.append(appt)
            labels.append(i % 3 == 0)
        predictor = CalibratedPredictor(base_predictor_cls=XGB)
        predictor.train(appts, labels)
        result = predictor.predict_no_show(appts[0])
        assert result.model_version == "calibrated-xgboost-v1"
```

- [ ] **Step 2: Implement CalibratedPredictor**

```python
# adapters/ml/calibrated_predictor.py
"""Calibrated predictor wrapper — isotonic calibration over XGBoost.

This is the project's identity (ADR-008). Calibrated probabilities
matter for ranking in the daily call list.
"""

from datetime import datetime
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.logistic_predictor import _score_to_category
from domain.models import Appointment, NoShowOutcome


class _SklearnBridge:
    """Bridge between our predictor interface and sklearn's estimator API.

    CalibratedClassifierCV needs an sklearn-compatible estimator with
    fit(), predict(), predict_proba(), get_params(), and classes_.
    """

    def __init__(self, base_predictor_cls: type) -> None:
        self._base = base_predictor_cls()
        self._encoder = FeatureEncoder()
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnBridge":
        # Reconstruct appointments not possible from X alone,
        # so we fit the raw sklearn model directly on X
        self._model = self._base._model.__class__(
            **self._base._model.get_params() if hasattr(self._base, "_model") and self._base._model else {}
        )
        from xgboost import XGBClassifier

        self._model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            scale_pos_weight=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"base_predictor_cls": type(self._base)}


class CalibratedPredictor:
    """Isotonic-calibrated predictor implementing NoShowPredictorPort.

    Wraps XGBoost (or any predictor) with sklearn CalibratedClassifierCV.
    """

    def __init__(self, base_predictor_cls: type | None = None) -> None:
        self._base_cls = base_predictor_cls
        self._calibrated_model: CalibratedClassifierCV | None = None
        self._encoder = FeatureEncoder()

    def train(
        self, appointments: list[Appointment], outcomes: list[bool]
    ) -> None:
        """Train base model + isotonic calibration."""
        X = self._encoder.fit_transform(appointments, outcomes)
        y = np.array([int(o) for o in outcomes])

        from xgboost import XGBClassifier

        base_model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            scale_pos_weight=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        n_folds = min(3, max(2, len(y) // 20))
        self._calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",
            cv=n_folds,
        )
        self._calibrated_model.fit(X, y)

    def predict_no_show(self, appointment: Appointment) -> NoShowOutcome:
        """Predict calibrated no-show probability."""
        if self._calibrated_model is None:
            raise RuntimeError(
                "Model must be trained before prediction. Call train() first."
            )
        X = self._encoder.transform([appointment])
        prob = float(self._calibrated_model.predict_proba(X)[0, 1])
        return NoShowOutcome(
            appointment_id=appointment.appointment_id,
            risk_score=prob,
            risk_category=_score_to_category(prob),
            assessment_timestamp=datetime.now(),
            model_version="calibrated-xgboost-v1",
        )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_predictors.py -v`
Expected: All 13 tests PASS (5 Logistic + 5 XGBoost + 3 Calibrated).

- [ ] **Step 4: Commit**

```bash
git add adapters/ml/calibrated_predictor.py tests/test_predictors.py
git commit -m "feat: CalibratedPredictor — isotonic calibration over XGBoost (project identity)"
```

---

## Task 7: Training Pipeline (Use Case Update)

**Files:**
- Modify: `application/use_cases.py`
- No separate test file — integration tested via Task 8

**Depends on:** Tasks 1–6

- [ ] **Step 1: Update use_cases.py with train_and_evaluate**

Replace contents of `application/use_cases.py`:

```python
# application/use_cases.py
"""Use cases: orchestration of domain and adapters."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from adapters.ml.evaluation import evaluate_model
from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Appointment, NoShowOutcome
from domain.ports import AppointmentRepository, NoShowPredictorPort


def predict_no_show(
    appointment: Appointment,
    predictor: NoShowPredictorPort,
) -> NoShowOutcome:
    """Orchestrate no-show prediction for one appointment."""
    return predictor.predict_no_show(appointment)


def train_and_evaluate(
    repository: AppointmentRepository,
    predictor_factories: dict[str, type],
    output_path: Path,
    k: int = 20,
) -> dict[str, dict]:
    """Train all models with GroupKFold CV + temporal holdout.

    Args:
        repository: Data source implementing AppointmentRepository.
        predictor_factories: Map of name → predictor class (e.g. LogisticPredictor).
        output_path: Path to save model_metrics.json.
        k: Top-K for precision@K.

    Returns:
        Nested dict of metrics per model per validation strategy.
    """
    # Load all data
    all_appointments = repository.get_appointments()
    all_labels = repository.get_labels()
    n_rows = len(all_appointments)
    n_patients = len({a.patient.patient_id for a in all_appointments})

    results: dict[str, dict] = {
        "dataset": "KaggleV2-May-2016.csv",
        "n_rows": n_rows,
        "n_patients": n_patients,
        "validation": {
            "groupkfold": {},
            "temporal_holdout": {},
        },
    }

    # === GroupKFold CV ===
    groups = np.array([a.patient.patient_id for a in all_appointments])
    y_all = np.array([int(lb) for lb in all_labels])
    gkf = GroupKFold(n_splits=5)

    for model_name, predictor_cls in predictor_factories.items():
        fold_metrics: list[dict] = []

        for train_idx, test_idx in gkf.split(
            np.zeros(n_rows), y_all, groups
        ):
            train_appts = [all_appointments[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            test_appts = [all_appointments[i] for i in test_idx]

            # Train predictor
            predictor = predictor_cls()
            predictor.train(train_appts, train_labels)

            # Predict on test fold
            y_prob = np.array(
                [
                    predictor.predict_no_show(appt).risk_score
                    for appt in test_appts
                ]
            )
            y_true = y_all[test_idx]

            metrics = evaluate_model(y_true, y_prob, model_name, k=k)
            fold_metrics.append(metrics)

        # Aggregate across folds
        metric_keys = ["auc", "f1", "brier", "precision_at_k", "ece"]
        aggregated: dict[str, str] = {}
        for key in metric_keys:
            values = [fm[key] for fm in fold_metrics]
            mean = float(np.mean(values))
            std = float(np.std(values))
            aggregated[key] = f"{mean:.4f} +/- {std:.4f}"

        results["validation"]["groupkfold"][model_name] = aggregated

    # === Temporal Holdout ===
    june_start = datetime(2016, 6, 1)
    train_appts = repository.get_appointments(end_date=june_start)
    train_labels = repository.get_labels(end_date=june_start)
    test_appts = repository.get_appointments(start_date=june_start)
    test_labels = repository.get_labels(start_date=june_start)

    if test_appts:
        y_test = np.array([int(lb) for lb in test_labels])

        for model_name, predictor_cls in predictor_factories.items():
            predictor = predictor_cls()
            predictor.train(train_appts, train_labels)

            y_prob = np.array(
                [
                    predictor.predict_no_show(appt).risk_score
                    for appt in test_appts
                ]
            )
            metrics = evaluate_model(y_test, y_prob, model_name, k=k)
            # Remove model_name from metrics dict for clean JSON
            metrics.pop("model_name", None)
            results["validation"]["temporal_holdout"][model_name] = metrics

        results["validation"]["temporal_holdout"]["train_period"] = (
            "2016-04 to 2016-05"
        )
        results["validation"]["temporal_holdout"]["test_period"] = "2016-06"

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))

    return results
```

- [ ] **Step 2: Commit**

```bash
git add application/use_cases.py
git commit -m "feat: train_and_evaluate use case — GroupKFold CV + temporal holdout"
```

---

## Task 8: Run Full Pipeline

**Files:**
- Create: `scripts/run_training.py`
- Output: `reports/model_metrics.json`

**Depends on:** All previous tasks

- [ ] **Step 1: Create training script**

```python
# scripts/run_training.py
"""Run the full training pipeline and save metrics."""

from pathlib import Path

from adapters.data.csv_repository import KaggleAppointmentCSVRepository
from adapters.ml.calibrated_predictor import CalibratedPredictor
from adapters.ml.logistic_predictor import LogisticPredictor
from adapters.ml.xgboost_predictor import XGBoostPredictor
from application.use_cases import train_and_evaluate

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
OUTPUT_PATH = ROOT / "reports" / "model_metrics.json"


def main() -> None:
    print(f"Loading data from {CSV_PATH}")
    repo = KaggleAppointmentCSVRepository(CSV_PATH)

    predictor_factories = {
        "logistic": LogisticPredictor,
        "xgboost": XGBoostPredictor,
        "calibrated_xgboost": CalibratedPredictor,
    }

    print("Training and evaluating models...")
    print("  - 5-fold GroupKFold CV")
    print("  - Temporal holdout (June 2016)")
    results = train_and_evaluate(
        repository=repo,
        predictor_factories=predictor_factories,
        output_path=OUTPUT_PATH,
        k=20,
    )

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("\n=== GroupKFold CV ===")
    for model, metrics in results["validation"]["groupkfold"].items():
        print(f"\n{model}:")
        for key, val in metrics.items():
            print(f"  {key}: {val}")

    if results["validation"]["temporal_holdout"]:
        print("\n=== Temporal Holdout (June) ===")
        for model, metrics in results["validation"][
            "temporal_holdout"
        ].items():
            if isinstance(metrics, dict):
                print(f"\n{model}:")
                for key, val in metrics.items():
                    print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the pipeline**

Run: `python scripts/run_training.py`
Expected: Prints metrics, saves `reports/model_metrics.json`.

This will take a few minutes on the full 110k-row dataset. Watch for:
- All 3 models produce results
- AUC should be ~0.65+ (matching EDA baseline)
- Calibrated XGBoost should have lower Brier score than uncalibrated

- [ ] **Step 3: Verify output**

Run: `cat reports/model_metrics.json | python -m json.tool`
Expected: Valid JSON with groupkfold and temporal_holdout sections for all 3 models.

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (26 existing + ~30 new = ~56 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_training.py reports/model_metrics.json
git commit -m "feat: full training pipeline — 3 models, GroupKFold + temporal holdout"
```

---

## Task 9: CI Verification + Final Cleanup

**Files:**
- Verify: all adapter `__init__.py` files have content
- Run: `make check` or equivalent

- [ ] **Step 1: Ensure adapter __init__.py files export correctly**

Verify `adapters/data/__init__.py` and `adapters/ml/__init__.py` exist (they do, but may be empty — that's fine for Protocol-based adapters).

- [ ] **Step 2: Run black + isort on all new files**

Run: `python -m black adapters/ tests/ scripts/ application/ && python -m isort --profile black adapters/ tests/ scripts/ application/`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A
git commit -m "style: format new adapter and test files"
```

- [ ] **Step 5: Push to dev and verify CI**

```bash
git push origin dev
```

Check: `gh run list --limit 3 --branch dev`
Expected: All 3 checks (Test, Lint, Security) pass ✅.
