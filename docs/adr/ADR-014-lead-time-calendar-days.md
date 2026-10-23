# ADR-014: Lead Time = Calendar-Day Difference; Drop Impossible Rows

**Date:** 2026-10-23
**Status:** Accepted (supersedes the lead-time handling in `csv_repository.py` and the "negative = data error" note in `reports/eda_gate.md`)
**Deciders:** Tirth Joshi

## Context

An audit before shipping found that `AppointmentDay` is always midnight while `ScheduledDay` includes a time of day. Lead time was computed by subtracting the raw timestamps and truncating to whole days:

| Real gap (calendar days) | Rows | No-show rate | Old stored value |
|---|---|---|---|
| Appointment before booking | 5 | n/a | 0 (clipped) |
| Same day | 38,563 | 4.6% | 0 (was -1, clipped) |
| Next day | 5,213 | 21.4% | **0** |
| 2+ days | 66,746 | 24 to 33% (29% overall) | **one day too low** |

The EDA gate had called the 34.9% "negative" lead times data errors. They were almost all same-day bookings, the strongest signal in the dataset, merged with next-day bookings that miss about 4.6 times as often.

**Plain English:** the model couldn't tell "booked this morning" from "booked yesterday", and every other gap was off by a day.

## Decision

- `lead_time_days = AppointmentDay.normalize() - ScheduledDay.normalize()` (calendar days).
- **Drop** the 5 rows where the appointment date is before the booking date. Don't clip them, or they pose as same-day.
- Regression tests: same-day = 0 and next-day = 1 are distinct; impossible rows are dropped; Apr 30 → May 5 = 5.
- Regenerate every artifact (training, Phase 2, demo).

## Consequences

Discrimination and CV calibration got slightly **better**, as expected for measurement error in the top feature. Nothing previously claimed was overstated. (June holdout ECE moved slightly the other way, 0.0156 → 0.0165, within noise.)

| Metric (calibrated XGBoost) | Before | After |
|---|---|---|
| 5-fold GroupKFold AUC | 0.7237 ± 0.0031 | **0.7311 ± 0.0042** |
| 5-fold ECE | 0.0070 ± 0.0011 | **0.0059 ± 0.0020** |
| Brier | 0.1452 | **0.1440** |
| Temporal holdout AUC (June) | 0.7158 | **0.7238** |
| Rows | 110,527 | 110,522 |

- Logistic baseline AUC 0.6565 → 0.6605.
- Global SHAP top 4 unchanged (lead time, age, neighbourhood, SMS); lead time's mean |SHAP| rose 0.84 → 0.94 now that same-day and next-day are separated. Ranks 5 to 10 reshuffled slightly.
- Lesson recorded in the decision tree: inspect timestamp precision before treating "impossible" values as errors.
