# ADR-011: Split Strategy — GroupKFold + Temporal Holdout

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

39% of patients have multiple appointments (max 88). Same patient in train + test = leakage. Dataset covers April 29 – June 8, 2016.

## Decision

Dual validation:
1. **5-fold GroupKFold by PatientId** : for model selection and tuning. Stable CV metrics.
2. **Temporal holdout (June)** — train April-May, test June. Validates temporal generalization.

Report both: "5-fold GroupKFold AUC: X ± Y" and "Temporal holdout AUC: Z"

## Consequences

- GroupKFold prevents patient leakage across folds
- Temporal split validates deployment realism (even with narrow window)
- Dual validation checks both patient leakage and time generalisation
- Target encoding (ADR-007) must be fold-aware in CV loop
