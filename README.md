# Patient Readmission Risk Engine

Production-grade ML system for predicting 30-day hospital readmission risk using credentialed EHR data. Built with Hexagonal Architecture and explicit data leakage prevention for clinical validity.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-6/6%20passing-success)](./tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Project Overview

This system predicts hospital readmission risk for patients at the point of discharge. The model uses only pre-discharge clinical features to ensure predictions reflect information genuinely available to clinicians at decision time. The baseline readmission rate in our cohort is approximately 17.5%, aligning with national benchmarks for 30-day unplanned readmissions.

The architecture enforces strict separation between domain logic (business rules, risk scoring) and infrastructure (data loading, ML models). This enables clinical validation independent of specific data sources or modeling frameworks.

---

## Architecture

**Hexagonal (Ports and Adapters) Design:**

```
patient-readmission-risk-engine/
├── domain/                 # Pure Python business logic
│   ├── models.py          # Patient, Encounter, RiskOutcome entities
│   ├── ports.py           # PatientDataRepository, RiskPredictorPort interfaces
│   ├── services.py        # Baseline risk assessment logic
│   └── exceptions.py      # DataLeakageError, domain-specific errors
├── adapters/              # External system connections
│   ├── data/              # CSV, database repository implementations
│   ├── ml/                # XGBoost, model training adapters
│   └── visualization/     # Streamlit, plotting adapters
├── application/           # Use case orchestration
│   └── use_cases.py       # train_model(), predict_readmission_risk()
└── tests/                 # Pytest suite with property-based tests
```

**Dependency Rule:** All dependencies point inward. The domain layer imports nothing from adapters or application. This makes the business logic testable in isolation and swappable (CSV to database, XGBoost to neural net) without touching domain code.

---

## Technology Stack

### Core Technologies
- **Python 3.12** - Type-safe, modern language features
- **XGBoost** - Gradient boosting for tabular clinical data
- **MLflow** - Experiment tracking, model versioning
- **SHAP** - Explainability for clinical stakeholders

### Data Science Tools
- **pandas** - EHR data manipulation
- **scikit-learn** - Preprocessing, evaluation metrics
- **Hypothesis** - Property-based testing for domain invariants

### Software Engineering Tools
- **pytest** - Test-driven development framework
- **Black** - Opinionated code formatting
- **Mypy** - Static type checking with `--strict` mode
- **Ruff** - Fast linting
- **pre-commit** - Automated quality gates

---

## Setup Instructions

### Prerequisites
- Python 3.12+
- Conda or Mamba (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd patient-readmission-risk-engine
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate patient-readmission-ml
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Verify setup:
```bash
pytest tests/ -v
```

Expected output: `6 passed in 0.01s`

---

## Data Leakage Prevention Protocol

**The Leakage Shield:** This project implements `DataLeakageError` as a domain-level exception that halts execution if post-discharge features are detected. This guarantees clinical validity.

**Prohibited Features:**
- Post-discharge medications
- Readmission diagnoses
- Actual length of stay (only *scheduled* LOS is permitted)
- Follow-up appointment data
- Discharge status

**Permitted Pre-Discharge Features:**
- Patient demographics (age, gender, insurance type)
- Admission type (Emergency, Elective, Urgent)
- Primary diagnosis at admission (ICD-10 code)
- Comorbidity count at admission
- Scheduled length of stay
- Prior admission count (historical data)

The `Encounter` domain model physically cannot contain post-discharge data. This structural guarantee prevents accidental leakage during feature engineering.

---

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=domain --cov=adapters --cov=application --cov-report=term-missing
```

Run property-based tests (requires `hypothesis`):
```bash
pytest tests/test_properties.py -v
```

---

## Project Status

**Current Phase:** Phase 2 - Integrity Audit Complete

| Milestone | Status |
|-----------|--------|
| Phase 1: Infrastructure & Hexagonal Architecture | ✅ Complete |
| Phase 2: Domain Models & Leakage Prevention | ✅ Complete |
| Phase 3: Adapter Implementation & ML Training | 🔄 In Progress |
| Phase 4: SHAP Explainability & Streamlit Dashboard | 📋 Planned |
| Phase 5: Cloud Deployment (AWS/Azure) | 📋 Planned |

---

## UBC MDS Alignment

This project demonstrates principles from the UBC Master of Data Science program:

| Concept | MDS Course |
|---------|------------|
| Hexagonal Architecture, TDD, reproducibility | DSCI 522 (Workflows) |
| Classification models, evaluation metrics | DSCI 571 (Supervised Learning I) |
| Feature selection, leakage prevention | DSCI 573 (Feature and Model Selection) |
| Clinical validity, stakeholder communication | DSCI 542 (Communication and Argumentation) |
| Hypothesis testing, statistical inference | DSCI 552/553 (Statistical Inference) |

---

## Business Impact

**Clinical Problem:** Hospital readmissions within 30 days of discharge are a key quality metric and cost driver. CMS penalizes hospitals with excess readmissions. Accurate risk prediction at discharge enables proactive interventions.

**Model Goal:** Flag high-risk patients for case management, discharge planning, and follow-up scheduling. A successful model reduces readmissions by 10-15%, saving $10,000+ per avoided readmission.

**Stakeholders:**
- Clinical: Discharge planners, case managers, attending physicians
- Operational: Hospital administrators, quality improvement teams
- Financial: Revenue cycle, value-based care contracts

---

## Contributing

This is a portfolio project demonstrating production-grade ML systems engineering for healthcare applications. For questions or collaboration, please open an issue.

---

## License

MIT License. See `LICENSE` file for details.

---

## Author

**Tirth Joshi**
UBC Master of Data Science
Former Data Systems Analyst, Vancouver General Hospital

Portfolio: [GitHub Profile]
LinkedIn: [Profile Link]
