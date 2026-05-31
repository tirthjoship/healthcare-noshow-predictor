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
- [x] KaggleAppointmentCSVRepository adapter (leakage guard, date filtering)
- [x] Logistic + XGBoost + Calibrated XGBoost trained
- [x] Metrics in `reports/model_metrics.json` (AUC, F1, Brier, ECE, precision@K)
- [x] GroupKFold (5-fold) + temporal holdout (June) results
- [x] README rebranded with disclaimer + real metrics
- [ ] Post-implementation review (feature sufficiency, thresholds)
- [ ] SHAP global + 3 local cases
- [ ] Fairness slice table in `reports/fairness.md`
- [ ] Streamlit demo (call list + drill-down)
- [ ] `docs/BUSINESS_IMPACT.md`

---

## 9. Phase 1 Results — Key Findings

### Model Metrics (5-fold GroupKFold CV)

| Model | AUC | Brier | ECE | Precision@20 |
|-------|-----|-------|-----|-------------|
| Logistic | 0.657 ± 0.005 | 0.155 ± 0.002 | 0.025 ± 0.002 | 0.38 ± 0.11 |
| XGBoost | 0.724 ± 0.003 | 0.214 ± 0.001 | 0.247 ± 0.005 | 0.54 ± 0.09 |
| **Calibrated XGBoost** | **0.724 ± 0.003** | **0.145 ± 0.002** | **0.007 ± 0.001** | **0.48 ± 0.08** |

### Key Observations

1. **Calibration identity confirmed** — same AUC, 32% lower Brier, 97% lower ECE
2. **XGBoost gains 10% AUC** over logistic (0.724 vs 0.657) — nonlinearity helps
3. **Precision@20 is moderate** (0.48-0.54) — call list will have ~50% hit rate
4. **Temporal holdout stable** — metrics consistent with CV (no temporal drift)
5. **Data fix:** 1 row with age=-1 clipped to 0 in adapter

### Flagged for Review (Phase 1.5)

- Feature sufficiency: 10 features may be enough (AUC 0.724 matches literature)
- Precision@20 variance is high (±0.08-0.11) — K=20 may be too small for stable estimates
- Calibrated XGBoost has slightly LOWER precision@20 than uncalibrated (0.48 vs 0.54) — investigate ranking vs calibration tradeoff
- Day-of-week and prior no-show count could add signal (not yet engineered)

---

## 10. Recalibrated Phase Plan

### Phase 1.5: Post-Implementation Review (optional, can merge into Phase 2)
- Review flagged items from Phase 1 findings
- Decide if feature engineering iteration needed before SHAP
- If precision@20 story is weak, adjust K or add features

### Phase 2: SHAP + Fairness (next session)
- SHAP global bar chart — "operations not clinical" narrative
- 3 local SHAP examples (young/long lead, elderly/same-day, SMS confound)
- Fairness: FPR/FNR + calibration per slice (Gender, Age band, Neighbourhood)
- `reports/fairness.md`

### Phase 3: Streamlit + Business Impact (final session)
- Streamlit: daily call list landing + SHAP drill-down
- `docs/BUSINESS_IMPACT.md` with real precision@K numbers
- Final README polish, remove any readmission references
- Update `career-ops/cv.md` with verified metrics

---

## 11. Session Log

### Session 1 (2026-05-30) — DONE
```text
Phase 0 EDA gate on data/raw/KaggleV2-May-2016.csv — all 5 gates PASS.
Domain pivot from readmission to no-show — 26 tests.
13 ADRs recorded via grill-me session.
Repo renamed to healthcare-noshow-predictor.
```

### Session 1 continued (2026-05-30) — DONE
```text
Phase 1 adapters + training pipeline.
6 adapter files, 4 test files, 65 tests green.
3 models: Logistic (AUC 0.657), XGBoost (0.724), Calibrated XGBoost (0.724, ECE 0.007).
Design spec + implementation plan written and executed via subagent-driven-development.
PR #3 merged to main. CI green (Test + Lint).
```

### Session 3 (next)
```text
Phase 2 SHAP + fairness report (reports/fairness.md).
```

### Session 4 — Docker + AWS S3 (see ../PORTFOLIO_TOOLS_PLAYBOOK.md)
```text
Add Dockerfile, docker-compose.yml, scripts/upload_artifacts.py (boto3, env AWS_PORTFOLIO_BUCKET).
Upload model + reports/model_metrics.json to S3 after train. README sections: Docker, AWS.
Optional: Lambda predict wrapper (P2). Skip EC2.
```

---

## 12. CV Update Rule

Only update `career-ops/cv.md` after Phase 2 fairness report exists.
Replace "Healthcare Readmission Risk Engine" with:
> Built calibrated no-show predictor (AUC 0.72, ECE 0.007) for clinic outreach teams; 65 tests, hexagonal architecture, GroupKFold + temporal validation.
