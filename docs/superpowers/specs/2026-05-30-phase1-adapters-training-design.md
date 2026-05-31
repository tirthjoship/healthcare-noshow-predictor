# Design Spec: Phase 1 — Adapters + Model Training

**Date:** 2026-05-30
**Author:** Tirth Joshi
**Status:** Approved (all decisions locked via grill-me session)
**ADRs:** 001–013 in `docs/adr/`

---

## 1. Goal

Build the adapter and training pipeline to produce a calibrated no-show prediction model. Outputs: trained models, `reports/model_metrics.json`, and a working `predict_no_show` use case.

**Not in scope:** SHAP, fairness, Streamlit, business impact doc (Phase 2).

---

## 2. Components

### 2.1 KaggleAppointmentCSVRepository

**Location:** `adapters/data/csv_repository.py`
**Implements:** `AppointmentRepository` (from `domain/ports.py`)

**Responsibilities:**
- Load `data/raw/KaggleV2-May-2016.csv`
- Parse `ScheduledDay` and `AppointmentDay` as datetime
- Compute `lead_time_days = (AppointmentDay - ScheduledDay).days`, clip to ≥ 0
- Map `No-show` → boolean label (Yes=True, No=False)
- Map `Hipertension` → `hypertension` (dataset typo)
- Construct `Patient` and `Appointment` domain objects
- Return labels via `get_labels()` aligned with `get_appointments()`

**Leakage guard:**
- Only columns knowable at scheduling time pass through
- `No-show` column used ONLY for labels, never as a feature
- Raise `DataLeakageError` if post-appointment columns detected

**Date filtering:**
- `start_date` / `end_date` filter on `AppointmentDay`
- Used by temporal holdout split (train April–May, test June)

### 2.2 FeatureEncoder

**Location:** `adapters/ml/feature_encoder.py`
**Purpose:** Transform `Appointment` objects into numeric feature matrix.

**Features (10):**

| Feature | Source | Type |
|---------|--------|------|
| `age` | Patient.age | int |
| `gender_enc` | Patient.gender == "F" → 1 | binary |
| `lead_time_days` | Appointment.lead_time_days | int |
| `scholarship` | Appointment.scholarship | binary |
| `hypertension` | Appointment.hypertension | binary |
| `diabetes` | Appointment.diabetes | binary |
| `alcoholism` | Appointment.alcoholism | binary |
| `handicap` | Appointment.handicap | ordinal (0–4) |
| `sms_received` | Appointment.sms_received | binary |
| `neighbourhood_enc` | Target-encoded neighbourhood | float |

**Target encoding (ADR-007):**
- `fit(appointments, labels)` — compute mean no-show rate per neighbourhood from training data only
- `transform(appointments)` — apply stored encoding; unseen neighbourhoods get global mean
- Must be re-fit per CV fold (fold-aware)
- Never fit on full dataset

**Interface:**
```python
class FeatureEncoder:
    def fit(self, appointments: list[Appointment], labels: list[bool]) -> None: ...
    def transform(self, appointments: list[Appointment]) -> np.ndarray: ...
    def fit_transform(self, appointments: list[Appointment], labels: list[bool]) -> np.ndarray: ...
    def get_feature_names(self) -> list[str]: ...
```

### 2.3 LogisticPredictor

**Location:** `adapters/ml/logistic_predictor.py`
**Implements:** `NoShowPredictorPort`

Wraps `sklearn.linear_model.LogisticRegression`:
- `train(appointments, outcomes)` — uses FeatureEncoder internally, fits logistic model
- `predict_no_show(appointment)` — returns `NoShowOutcome` with probability and category
- Category thresholds: < 0.2 = Low Risk, 0.2–0.4 = Medium Risk, > 0.4 = High Risk
- `model_version = "logistic-v1"`

### 2.4 XGBoostPredictor

**Location:** `adapters/ml/xgboost_predictor.py`
**Implements:** `NoShowPredictorPort`

Wraps `xgboost.XGBClassifier`:
- Same interface as LogisticPredictor
- Hyperparameters: `max_depth=4, n_estimators=200, learning_rate=0.1, scale_pos_weight=4` (class imbalance ~80/20)
- `model_version = "xgboost-v1"`

### 2.5 CalibratedPredictor

**Location:** `adapters/ml/calibrated_predictor.py`
**Implements:** `NoShowPredictorPort`

Wraps any `NoShowPredictorPort` with isotonic calibration:
- `train(appointments, outcomes)` — fits base model, then fits `CalibratedClassifierCV(method='isotonic', cv=5)` on held-out fold predictions
- `predict_no_show(appointment)` — returns calibrated probabilities
- `model_version = "calibrated-xgboost-v1"`

This is the project's identity (ADR-008).

### 2.6 ModelEvaluator

**Location:** `adapters/ml/evaluation.py`

Pure function, not a port — computes metrics from predictions:

```python
def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    k: int = 20,
) -> dict[str, float]:
    """Compute AUC, F1, Brier score, precision@K, calibration error."""
```

**Metrics (ADR-008):**
- AUC-ROC
- F1 (at optimal threshold)
- Brier score (calibration quality)
- Precision@K (for top-K call list, ADR-005)
- Expected Calibration Error (ECE, 10 bins)

Output saved to `reports/model_metrics.json`.

### 2.7 Training Pipeline (Use Case)

**Location:** `application/use_cases.py` — update existing `train_model()`

**Flow:**
1. Load data via `AppointmentRepository`
2. Split: GroupKFold (5-fold, by PatientId) for CV metrics
3. For each fold:
   - Fit `FeatureEncoder` on train fold only
   - Train model (Logistic / XGBoost / Calibrated)
   - Predict on test fold
   - Collect predictions
4. Compute CV metrics (mean ± std across folds)
5. Temporal holdout: train April–May, test June
6. Save all metrics to `reports/model_metrics.json`

**New use case:**

```python
def train_and_evaluate(
    repository: AppointmentRepository,
    predictors: dict[str, NoShowPredictorPort],
    k: int = 20,
) -> dict[str, dict[str, float]]:
    """Train all models, return metrics per model."""
```

---

## 3. File Layout (new files)

```
adapters/
├── data/
│   ├── __init__.py
│   └── csv_repository.py          # KaggleAppointmentCSVRepository
├── ml/
│   ├── __init__.py
│   ├── feature_encoder.py          # FeatureEncoder (target encoding)
│   ├── logistic_predictor.py       # LogisticPredictor
│   ├── xgboost_predictor.py        # XGBoostPredictor
│   ├── calibrated_predictor.py     # CalibratedPredictor
│   └── evaluation.py               # evaluate_model()
application/
└── use_cases.py                     # Updated: train_and_evaluate()
reports/
└── model_metrics.json               # Output
tests/
├── test_csv_repository.py           # Adapter: load, leakage, date filter
├── test_feature_encoder.py          # Encoding, leakage, fold-awareness
├── test_predictors.py               # Train/predict on small fixture
└── test_evaluation.py               # Metric computation
```

---

## 4. Validation Strategy (ADR-011)

### 4.1 GroupKFold (primary)

- 5 folds, grouped by `PatientId`
- Report: mean ± std for AUC, F1, Brier, precision@K
- FeatureEncoder re-fit per fold (no target leakage)

### 4.2 Temporal Holdout (secondary)

- Train: `AppointmentDay` in April–May 2016
- Test: `AppointmentDay` in June 2016
- Single point estimate for each metric
- Validates deployment realism

---

## 5. Risk Categorization

| Score Range | Category | Action |
|-------------|----------|--------|
| < 0.20 | Low Risk | No outreach |
| 0.20 – 0.40 | Medium Risk | SMS reminder |
| > 0.40 | High Risk | Phone call |

These thresholds are initial. Top-K ranking (ADR-005) overrides fixed thresholds for the daily call list.

---

## 6. Data Flow Diagram

```
KaggleV2-May-2016.csv
        │
        ▼
KaggleAppointmentCSVRepository
  ├── get_appointments() → list[Appointment]
  └── get_labels() → list[bool]
        │
        ▼
  GroupKFold split (by PatientId)
        │
   ┌────┴────┐
   train     test
   │         │
   ▼         │
FeatureEncoder.fit()
   │         │
   ▼         ▼
FeatureEncoder.transform()
   │         │
   ▼         ▼
Predictor.train()
   │         │
   │         ▼
   │    Predictor.predict_no_show()
   │         │
   │         ▼
   │    evaluate_model()
   │         │
   ▼         ▼
reports/model_metrics.json
```

---

## 7. Test Strategy

### Unit Tests (small fixtures, no full dataset)

| Test File | What It Tests |
|-----------|---------------|
| `test_csv_repository.py` | CSV loading, date parsing, lead_time computation, leakage guard, date filtering |
| `test_feature_encoder.py` | Feature vector shape, target encoding on train only, unseen neighbourhood fallback, feature names |
| `test_predictors.py` | Logistic/XGBoost/Calibrated train + predict on 50-row fixture, output is valid NoShowOutcome |
| `test_evaluation.py` | Metric computation (AUC, F1, Brier, precision@K, ECE) on known arrays |

### Property Tests (Hypothesis)

| Property | File |
|----------|------|
| FeatureEncoder always produces 10 columns | `test_feature_encoder.py` |
| All predictors return score in [0, 1] | `test_predictors.py` |
| Evaluation metrics are finite and bounded | `test_evaluation.py` |

### Integration Test

| Test | File |
|------|------|
| Full pipeline: CSV → encode → train → predict → evaluate | `test_csv_repository.py` (small CSV fixture via `tmp_path`) |

---

## 8. Output Artifacts

### `reports/model_metrics.json`

```json
{
  "dataset": "KaggleV2-May-2016.csv",
  "n_rows": 110527,
  "n_patients": 62299,
  "validation": {
    "groupkfold": {
      "n_folds": 5,
      "logistic": {"auc": "0.XX ± 0.XX", "f1": "...", "brier": "...", "precision_at_20": "..."},
      "xgboost": {"auc": "...", "f1": "...", "brier": "...", "precision_at_20": "..."},
      "calibrated_xgboost": {"auc": "...", "f1": "...", "brier": "...", "precision_at_20": "..."}
    },
    "temporal_holdout": {
      "train_period": "2016-04 to 2016-05",
      "test_period": "2016-06",
      "logistic": {"auc": "...", "f1": "...", "brier": "...", "precision_at_20": "..."},
      "xgboost": {"auc": "...", "f1": "...", "brier": "...", "precision_at_20": "..."},
      "calibrated_xgboost": {"auc": "...", "f1": "...", "brier": "...", "precision_at_20": "..."}
    }
  }
}
```

---

## 9. Dependencies

Already in `pyproject.toml`:
- `pandas` — CSV loading
- `scikit-learn` — LogisticRegression, GroupKFold, CalibratedClassifierCV, metrics
- `xgboost` — XGBClassifier
- `numpy` — feature arrays

No new dependencies required.

---

## 10. Success Criteria

- [ ] `KaggleAppointmentCSVRepository` loads CSV, returns domain objects, guards leakage
- [ ] `FeatureEncoder` produces 10-column matrix, target-encodes neighbourhood train-only
- [ ] 3 predictors train and produce valid `NoShowOutcome` objects
- [ ] GroupKFold CV metrics computed (AUC, F1, Brier, precision@K)
- [ ] Temporal holdout metrics computed
- [ ] `reports/model_metrics.json` saved with all metrics
- [ ] All existing + new tests pass
- [ ] CI green (lint + test + security)
- [ ] Calibrated XGBoost shows lower Brier score than uncalibrated (project identity)

---

## 11. Implementation Order

1. `csv_repository.py` + `test_csv_repository.py` — data foundation
2. `feature_encoder.py` + `test_feature_encoder.py` — encoding pipeline
3. `logistic_predictor.py` + `test_predictors.py` — baseline model
4. `xgboost_predictor.py` — extend test_predictors
5. `evaluation.py` + `test_evaluation.py` — metrics
6. `calibrated_predictor.py` — extend test_predictors
7. `use_cases.py` update — training pipeline
8. Run full pipeline → `reports/model_metrics.json`
9. CI verification

**Estimated subagent tasks:** 4–5 independent Sonnet subagents (steps 1–2 parallel, 3–5 parallel, 6–8 sequential).
