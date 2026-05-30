# CONTEXT.md — Healthcare Appointment No-Show Predictor

**Repo:** `healthcare-noshow-predictor` (local folder: `patient-readmission-risk-engine` — rename pending)
**Remote:** `https://github.com/tirthjoship/healthcare-noshow-predictor.git`
**Owner:** Tirth Joshi
**Created:** 2026-05-30
**Phase:** 1 — adapters + training (EDA gate passed)
**Portfolio slot:** Project 3 of 5

**Read first:** [`../PORTFOLIO_LOCKED_DECISIONS.md`](../PORTFOLIO_LOCKED_DECISIONS.md) · [`../PORTFOLIO_EDA_SPRINT.md`](../PORTFOLIO_EDA_SPRINT.md) · [`docs/adr/`](docs/adr/)

---

## 1. Mission

Predict **medical appointment no-shows** at booking time so clinic outreach teams can target reminders, recover appointment slots, and reduce revenue loss.

**Pivot rationale:** MIMIC-IV readmission blocked (no credentials). No-show prediction uses public data, clear ROI, and matches **VGH operations / outreach** narrative without claiming employer data.

---

## 2. Locked Decisions (13 ADRs)

| # | Decision | Choice | ADR |
|---|----------|--------|-----|
| 1 | Problem pivot | No-show prediction (from readmission) | ADR-001 |
| 2 | Primary persona | Outreach / patient engagement team | ADR-002 |
| 3 | Intervention model | Daily call list (morning huddle) | ADR-003 |
| 4 | Business impact | Revenue-based, $200/slot, assumptions labeled | ADR-004 |
| 5 | Threshold strategy | Capacity-constrained top-K | ADR-005 |
| 6 | SMS confound | Include + prominent documentation (Simpson's paradox) | ADR-006 |
| 7 | Neighbourhood encoding | Target encoding, train-only, fold-aware | ADR-007 |
| 8 | Model lineup | Logistic → XGBoost → Calibrated XGBoost | ADR-008 |
| 9 | Fairness reporting | FPR/FNR + calibration per demographic slice | ADR-009 |
| 10 | Streamlit demo | Call list landing + SHAP drill-down (2 pages) | ADR-010 |
| 11 | Split strategy | GroupKFold (CV) + temporal holdout (June) | ADR-011 |
| 12 | SHAP narrative | "Operations problem, not clinical" | ADR-012 |
| 13 | Repo name | `healthcare-noshow-predictor` | ADR-013 |

Additional locked decisions:
- **Dataset:** [Kaggle Medical Appointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments) — `KaggleV2-May-2016.csv`
- **Architecture:** Hexagonal (ports & adapters)
- **NOT using:** MIMIC-IV, UCI readmission, VGH/BCCNM production data
- **Evaluation:** AUC, F1, Brier score, calibration curve, precision@K

---

## 3. Dataset Schema

| Column | Feature use |
|--------|-------------|
| `PatientId` | Grouped split (39% have multiple appointments) |
| `AppointmentID` | Row id |
| `Gender` | Feature (M/F) |
| `ScheduledDay` | Parse datetime → derive lead_time |
| `AppointmentDay` | Parse datetime |
| `Age` | Feature |
| `Neighbourhood` | Feature (81 unique — target encode on train only) |
| `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` | Binary/ordinal features |
| `SMS_received` | Feature (intervention confound — document Simpson's paradox) |
| `No-show` | Target (`Yes`/`No` → 1/0) |

**Engineered feature:** `lead_time_days = (AppointmentDay - ScheduledDay).days` (clipped ≥ 0)

**Leakage rule:** All features must be knowable at scheduling time. No post-appointment data.

---

## 4. EDA Gate Results (PASSED)

| Gate | Criterion | Result |
|------|-----------|--------|
| Target rate 15-30% | 20.2% | ✅ |
| Logistic AUC ≥ 0.65 | 0.6558 | ✅ |
| No single feature >80% | max 32.9% | ✅ |
| No missing values | 0 nulls | ✅ |
| Sufficient volume | 110,527 rows | ✅ |

Key findings: `reports/eda_gate.md`

---

## 5. Project Identity (What Makes This Unique)

- **Calibration focus** — Logistic → XGBoost → Calibrated XGBoost. Probability quality matters for ranking.
- **SHAP comparison narrative** — "This is an operations problem, not a clinical one" (scheduling features >> clinical features)
- **SMS Simpson's paradox** — documented confound, strong interview talking point
- **Dual validation** — GroupKFold + temporal holdout
- **Fairness** — FPR/FNR + calibration per demographic slice

---

## 6. Domain Model (Pivoted)

| Old (readmission) | New (no-show) |
|-------------------|---------------|
| `Encounter` | `Appointment` |
| `Patient` | `Patient` (slimmed: removed insurance_type) |
| `RiskOutcome` | `NoShowOutcome` |
| `PatientDataRepository` | `AppointmentRepository` |
| `RiskPredictorPort` | `NoShowPredictorPort` |
| `InvalidAdmissionDataError` | `InvalidAppointmentDataError` |
| MIMIC adapter | `KaggleAppointmentCSVRepository` |
| Clinical SHAP narrative | Ops narrative: lead time, age, SMS |

---

## 7. Business Impact Model

```
recoverable_slots = top_K_count × precision_at_K
cost_per_noshow = $200 (primary care literature estimate, assumption labeled)
monthly_value = recoverable_slots × cost_per_noshow × 22 working_days
```

Output: `docs/BUSINESS_IMPACT.md`

---

## 8. Success Criteria

- [x] EDA gate GO
- [ ] KaggleAppointmentCSVRepository adapter
- [ ] Logistic + XGBoost + Calibrated XGBoost trained
- [ ] Metrics in `reports/model_metrics.json` (AUC, F1, Brier, calibration)
- [ ] GroupKFold + temporal holdout results
- [ ] SHAP global + 3 local cases
- [ ] Fairness slice table in `reports/fairness.md`
- [ ] Streamlit demo (call list + drill-down)
- [ ] `docs/BUSINESS_IMPACT.md`
- [ ] README rebranded with disclaimer

---

## 9. Session Playbook

### Session 1 (DONE)
```text
Phase 0 EDA gate on data/raw/KaggleV2-May-2016.csv.
Domain pivot from readmission to no-show.
13 ADRs recorded. CONTEXT.md updated.
```

### Session 2
```text
Build KaggleAppointmentCSVRepository adapter.
Train Logistic + XGBoost + Calibrated XGBoost.
GroupKFold CV + temporal holdout. Save reports/model_metrics.json.
```

### Session 3
```text
SHAP (global + 3 local), fairness report, Streamlit demo.
docs/BUSINESS_IMPACT.md, README rebrand.
Remove readmission references unless marked future work.
```

---

## 10. CV Update Rule

Only update `career-ops/cv.md` after metrics exist in `reports/`.
Replace "Healthcare Readmission Risk Engine" bullet with no-show wording + real AUC.
