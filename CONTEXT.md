# CONTEXT.md — Healthcare Appointment No-Show Risk (Pivot)

**Repo:** `patient-readmission-risk-engine` (folder name unchanged; **rebrand in README**)  
**Owner:** Tirth Joshi  
**Created:** 2026-05-30  
**Phase:** 0 — EDA gate → pivot adapters → train  
**Portfolio slot:** Project 3 of 5

**Read first:** [`../PORTFOLIO_LOCKED_DECISIONS.md`](../PORTFOLIO_LOCKED_DECISIONS.md) · [`../PORTFOLIO_EDA_SPRINT.md`](../PORTFOLIO_EDA_SPRINT.md)

---

## 1. Mission

Predict **medical appointment no-shows** at booking time so clinics can target reminders, overbook strategically, or reallocate capacity.

**Pivot rationale:** MIMIC-IV readmission blocked (no credentials). No-show prediction uses public data, clear ROI, and matches **VGH operations / outreach** narrative without claiming employer data.

---

## 2. Locked decisions

| Decision | Choice |
|----------|--------|
| **Problem** | Binary no-show classification |
| **Dataset** | [Kaggle Medical Appointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments) |
| **File** | `KaggleV2-May-2016.csv` → `data/raw/` |
| **Architecture** | Keep existing **hexagonal** layout; replace readmission domain with appointment domain |
| **NOT using** | MIMIC-IV, UCI readmission (v1), VGH/BCCNM production data |
| **Evaluation** | AUC, F1, calibration, SHAP; compare to rule baseline |
| **Fairness** | Slices: gender, age band, top neighbourhoods |
| **Demo** | Streamlit with disclaimer banner |

---

## 3. Dataset schema

| Column | Feature use |
|--------|-------------|
| `PatientId` | Grouped split if repeat patients |
| `AppointmentID` | Row id |
| `Gender` | Feature |
| `ScheduledDay` | Parse datetime |
| `AppointmentDay` | Parse datetime |
| `Age` | Feature |
| `Neighbourhood` | Feature (high cardinality — target encode on train only) |
| `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` | Binary features |
| `SMS_received` | Feature (prior intervention — document confound) |
| `No-show` | Target (`Yes`/`No` → 1/0) |

**Engineered feature:** `lead_time_days = (AppointmentDay - ScheduledDay).days`

**Leakage rule:** No post-appointment columns. All features knowable at schedule time.

---

## 4. Expected signal (literature sanity check)

- National/clinic no-show rates ~15–30%; dataset ~20%
- Literature AUC ~0.70–0.75 with similar features
- If EDA shows AUC < 0.60 on logistic baseline → investigate data issues before proceeding

---

## 5. Phase 0 — EDA gate (START HERE)

Outputs:
- `notebooks/00_eda_gate.ipynb`
- `reports/eda_gate.md`

See full checklist in `../PORTFOLIO_EDA_SPRINT.md` § Project 3.

---

## 6. Pivot implementation map

| Readmission (old) | No-show (new) |
|-------------------|---------------|
| `Encounter` | `Appointment` |
| `Patient` | `Patient` (keep) |
| `RiskOutcome` | `NoShowOutcome` |
| `readmission` label | `no_show` label |
| `DataLeakageError` | Keep — post-appointment fields forbidden |
| MIMIC adapter | `KaggleAppointmentCSVRepository` |
| Clinical SHAP narrative | Ops narrative: SMS, lead time, age |

**Rename in README/title** to `Healthcare Appointment No-Show Risk Engine`.  
**Defer** folder rename on GitHub until user approves remote URL change.

---

## 7. Business impact doc

`docs/BUSINESS_IMPACT.md`:
```
recoverable_slots = top_decile_count × (precision at threshold)
cost_per_noshow = $X (label assumption)
monthly_value = recoverable_slots × cost_per_noshow
```

---

## 8. Success criteria

- EDA gate GO
- Logistic + XGBoost trained; metrics in `reports/model_metrics.json`
- SHAP global + 3 local cases in docs
- Fairness slice table in `reports/fairness.md`
- Streamlit demo
- README disclaimer: public benchmark data

---

## 9. Claude Code — session playbook

### Session 1
```text
Read CONTEXT.md. Phase 0 EDA only on data/raw/KaggleV2-May-2016.csv.
Produce notebooks/00_eda_gate.ipynb and reports/eda_gate.md.
Do not refactor domain yet if EDA fails gate.
```

### Session 2
```text
EDA passed. Pivot domain models and adapters from readmission to no-show.
Implement KaggleAppointmentCSVRepository. Update tests.
Train logistic + XGBoost; save reports/model_metrics.json.
```

### Session 3
```text
SHAP, fairness report, Streamlit, README rebrand, docs/BUSINESS_IMPACT.md.
Remove MIMIC references from README unless marked future work.
```

---

## 10. CV update rule

Only update `career-ops/cv.md` after metrics exist in `reports/`.  
Replace "Healthcare Readmission Risk Engine" bullet with no-show wording + real AUC.
