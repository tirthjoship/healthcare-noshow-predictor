# EDA Gate Report — Healthcare Appointment No-Show

**Date:** 2026-05-30
**Dataset:** `KaggleV2-May-2016.csv`
**Rows:** 110,527 | **Columns:** 19

---

## 1. Schema & Data Quality

- **14 columns**, all expected present
- **Zero nulls** across all columns
- Dtypes: 2 datetime (parsed), 5 binary, 1 categorical (Neighbourhood), 1 continuous (Age)
- Negative lead_time rows: 38568 (34.89%) — clipped to 0

## 2. Target Distribution

- **No-show rate: 20.2%** (within expected 15-30% range)
- Class imbalance ~80/20 — moderate, manageable with stratification

## 3. Feature Analysis

### Lead Time
- Mean: 9.5 days, Median: 3 days
- Strong signal: no-show rate increases with lead time
- Same-day appointments have lowest no-show rate

### SMS Received (CONFOUND)
- **SMS_received=1 has HIGHER no-show rate** — Simpson's paradox
- SMS recipients have longer lead times (mean 18.0 vs 5.5 days)
- SMS is sent as intervention to high-risk patients → post-hoc feature
- **Decision:** Include with documented caveat. Model should not be interpreted as "SMS causes no-shows"

### Demographics
- Gender: minimal difference in no-show rates
- Age: younger patients (19-35) have highest no-show rates
- Neighbourhood: 81 unique values, top 15 cover majority of volume

### Duplicate Patients
- 62,299 unique patients across 110,527 appointments
- 24,379 patients (39.1%) have multiple appointments
- **Must use GroupShuffleSplit** to prevent data leakage across train/test

## 4. Baseline Model

- **Logistic Regression AUC: 0.6558** (grouped holdout, 80/20)
- Features: Age, lead_time, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, Gender, Neighbourhood (rank-encoded)

### Feature Importance (|coefficient|)

| Feature | |Coef| | % of Total |
|---------|--------|------------|
| SMS_received | 0.3559 | 32.9% |
| Scholarship | 0.2257 | 20.9% |
| Alcoholism | 0.1803 | 16.7% |
| Diabetes | 0.1267 | 11.7% |
| Hipertension | 0.0795 | 7.3% |
| Handcap | 0.0460 | 4.2% |
| Gender_enc | 0.0316 | 2.9% |
| lead_time_days | 0.0232 | 2.1% |
| Age | 0.0073 | 0.7% |
| neigh_rank | 0.0062 | 0.6% |

- No single feature dominates (max 32.9%) — healthy signal distribution

## 5. Gate Verdicts

| Gate | Criterion | Value | Status |
|------|-----------|-------|--------|
| 1. Target rate 15-30% | — | 20.2% | PASS ✅ |
| 2. Logistic AUC >= 0.65 | — | 0.6558 | PASS ✅ |
| 3. No single feature >80% importance | — | max 32.9% | PASS ✅ |
| 4. No missing values | — | 0 nulls (raw columns) | PASS ✅ |
| 5. Sufficient volume (>=50k) | — | 110,527 rows | PASS ✅ |

## Overall Verdict: **GO** 🟢

Proceed to Phase 1: domain model pivot and model training.

## Risks & Notes

1. **SMS confound** — document in model card; do not interpret SMS coefficient causally
2. **Grouped split required** — repeat patients must stay in same fold
3. **Neighbourhood encoding** — use target encoding on train only to prevent leakage
4. **Lead time** is strongest univariate signal — expected and healthy
5. **Age outliers** — some negative ages exist (likely data entry errors); minimal count
