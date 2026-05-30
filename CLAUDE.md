# CLAUDE.md

This file provides guidance to Claude Code when working with this repository. Read `AGENTS.md` for coding standards.

## Project Context

**Healthcare Appointment No-Show Predictor** — predicts no-shows at booking time for clinic outreach teams.

Read **`CONTEXT.md`** for all 13 locked decisions (ADRs), dataset schema, EDA results, and session playbook.

Public benchmark only — Kaggle Medical Appointments dataset. Not VGH/BCCNM production data.

## Architecture

Hexagonal (Ports & Adapters) with inward-pointing dependencies.
- domain/ — pure business logic, zero external imports
- adapters/ — data sources, ML models, visualization
- application/ — orchestration (composition root)

## Key Domain Objects

- `Patient` — immutable, patient_id + age + gender
- `Appointment` — immutable, 12 fields knowable at scheduling time
- `NoShowOutcome` — prediction result with score, category, SHAP explanation
- `AppointmentRepository` — port for data loading (runtime_checkable Protocol)
- `NoShowPredictorPort` — port for ML models

## Leakage Rule

All features must be knowable at **scheduling time**. Post-appointment data = `DataLeakageError`. Target encoding fit on train only (ADR-007).

## SMS Confound

SMS_received=1 has higher no-show rate (Simpson's paradox). SMS is intervention, not cause. Include as feature but document confound prominently (ADR-006).

## Commands

```bash
make check    # lint + typecheck + test with coverage
make test     # pytest -v --tb=short
make lint     # pre-commit run --all-files
make typecheck # mypy strict
```

## Phase Status

- [x] Phase 0: EDA gate — PASSED (AUC 0.6558, 20.2% no-show rate)
- [x] Phase 0.5: Domain pivot — Appointment/NoShowOutcome, 18 tests green
- [ ] Phase 1: Adapters + model training
- [ ] Phase 2: SHAP, fairness, Streamlit, business impact
