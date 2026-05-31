# Healthcare Appointment No-Show Predictor

Predict medical appointment no-shows at booking time so clinic outreach teams can target reminders, recover slots, and reduce revenue loss. Built with Hexagonal Architecture, calibration-focused modeling, and explicit data leakage prevention.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-success)](./tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Disclaimer:** This project uses the [Kaggle Medical Appointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments) public benchmark dataset. It is not connected to any health system or employer data.

---

## Project Overview

An outreach team at a clinic wants to reduce appointment no-shows. Each morning, they need a ranked **daily call list** of patients most likely to miss their upcoming appointments, so they can proactively call and confirm or reschedule.

This system:
1. Predicts no-show probability using only **scheduling-time features** (no post-appointment data)
2. Ranks patients by risk → generates a **capacity-constrained top-K call list**
3. Explains each prediction with **SHAP** — framed as an operations problem, not a clinical one
4. Reports **fairness metrics** (FPR/FNR + calibration) across demographic slices

### Key Finding

> **"This is an operations problem, not a clinical one."** Lead time and age dominate predictions. Clinical features (hypertension, diabetes, handicap) contribute minimally. Scheduling practices drive no-shows more than patient health conditions.

---

## Architecture

**Hexagonal (Ports & Adapters):**

```
healthcare-noshow-predictor/
├── domain/                 # Pure Python — zero external imports
│   ├── models.py          # Patient, Appointment, NoShowOutcome
│   ├── ports.py           # AppointmentRepository, NoShowPredictorPort
│   ├── services.py        # Baseline no-show risk heuristic
│   └── exceptions.py      # DataLeakageError, domain errors
├── adapters/              # External connections
│   ├── data/              # KaggleAppointmentCSVRepository
│   ├── ml/                # Logistic, XGBoost, CalibratedXGBoost
│   └── visualization/     # Streamlit components
├── application/           # Orchestration (composition root)
│   └── use_cases.py       # train_model(), predict_no_show()
├── tests/                 # Unit + property-based (Hypothesis)
├── notebooks/             # EDA only — no production logic
├── docs/adr/              # 13 Architecture Decision Records
└── reports/               # EDA gate, model metrics, fairness
```

**Dependency rule:** All dependencies point inward. Domain imports nothing from adapters or application.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle Medical Appointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments) |
| File | `KaggleV2-May-2016.csv` |
| Rows | 110,527 appointments |
| Patients | 62,299 unique (39% have multiple appointments) |
| Target | No-show rate: **20.2%** |
| Period | April 29 – June 8, 2016 |

**Leakage rule:** All features must be knowable at scheduling time. `DataLeakageError` raised if post-appointment features detected.

**SMS confound:** SMS_received=1 correlates with *higher* no-show rate (Simpson's paradox). SMS is sent as intervention to high-risk patients with longer lead times. Included as feature with documented caveat — not a causal predictor. See [ADR-006](docs/adr/ADR-006-sms-confound.md).

---

## EDA Gate Results

| Gate | Criterion | Result | Status |
|------|-----------|--------|--------|
| Target rate | 15–30% | 20.2% | ✅ |
| Logistic AUC | ≥ 0.65 | 0.6558 | ✅ |
| Feature dominance | No single >80% | max 32.9% | ✅ |
| Missing values | 0 | 0 | ✅ |
| Volume | ≥ 50k | 110,527 | ✅ |

Full report: [`reports/eda_gate.md`](reports/eda_gate.md)

---

## Model Lineup

| Model | Purpose | Identity |
|-------|---------|----------|
| Logistic Regression | Linear baseline | Interpretable, calibrated by default |
| XGBoost | Nonlinear gains | How much does tree-based modeling buy? |
| Calibrated XGBoost | Probability quality | **This project's differentiator** — isotonic calibration for ranked call lists |

**Evaluation:** AUC, F1, Brier score, calibration curve, precision@K

**Validation:** 5-fold GroupKFold (by PatientId) + temporal holdout (June). See [ADR-011](docs/adr/ADR-011-split-strategy.md).

---

## Business Impact Model

```
recoverable_slots = top_K × precision_at_K
cost_per_noshow   = $200 (primary care literature estimate — assumption labeled)
monthly_value     = recoverable_slots × $200 × 22 working_days
```

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12+ |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Testing | pytest, Hypothesis (property-based) |
| Quality | black, isort, mypy (strict), ruff, pre-commit |
| CI | GitHub Actions (lint, test, security) |

---

## Setup

```bash
git clone https://github.com/tirthjoship/healthcare-noshow-predictor.git
cd healthcare-noshow-predictor
pip install -e ".[dev]"
pre-commit install

# Place dataset
# Download from Kaggle → data/raw/KaggleV2-May-2016.csv

# Run tests
make test

# Full quality check
make check
```

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | EDA gate — dataset validation | ✅ PASSED |
| 0.5 | Domain pivot — readmission → no-show | ✅ Complete (18 tests) |
| 1 | Adapters + model training | 🔄 Next |
| 2 | SHAP, fairness, Streamlit, business impact | 📋 Planned |

---

## Architecture Decision Records

13 ADRs in [`docs/adr/`](docs/adr/):

| ADR | Decision |
|-----|----------|
| [001](docs/adr/ADR-001-pivot-to-noshow.md) | Pivot from readmission to no-show |
| [002](docs/adr/ADR-002-primary-persona.md) | Primary persona: outreach team |
| [003](docs/adr/ADR-003-daily-call-list.md) | Daily call list intervention |
| [004](docs/adr/ADR-004-revenue-based-impact.md) | Revenue-based impact ($200/slot) |
| [005](docs/adr/ADR-005-capacity-constrained-topk.md) | Capacity-constrained top-K threshold |
| [006](docs/adr/ADR-006-sms-confound.md) | SMS confound documentation |
| [007](docs/adr/ADR-007-target-encoding.md) | Target encoding (train-only) |
| [008](docs/adr/ADR-008-model-lineup.md) | Logistic → XGBoost → Calibrated XGBoost |
| [009](docs/adr/ADR-009-fairness-reporting.md) | FPR/FNR + calibration per slice |
| [010](docs/adr/ADR-010-streamlit-demo.md) | Call list + SHAP drill-down demo |
| [011](docs/adr/ADR-011-split-strategy.md) | GroupKFold + temporal holdout |
| [012](docs/adr/ADR-012-shap-narrative.md) | "Operations, not clinical" framing |
| [013](docs/adr/ADR-013-repo-rename.md) | Repo rename to healthcare-noshow-predictor |

---

## Limitations

- **Public benchmark data** — not production clinical data
- **SMS confound** — SMS_received coefficient is not causal (Simpson's paradox)
- **Narrow time window** — dataset covers ~6 weeks (April–June 2016)
- **Single clinic system** — Vitória, Brazil; may not generalize to other geographies
- **No cost optimization** — threshold is capacity-based, not cost-optimized

---

## Author

**Tirth Joshi**
UBC Master of Data Science
Former Data Systems Analyst, Vancouver General Hospital

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.
