# ADR-008: Model Lineup — Logistic → XGBoost → Calibrated XGBoost

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Need model comparison narrative that differentiates this project from supply-chain (Logistic vs XGBoost) and stock recommender (stacking ensemble).

## Decision

Three-stage progression:
1. **Logistic Regression** — linear baseline, interpretable
2. **XGBoost** — nonlinear, measures how much tree-based gains
3. **Calibrated XGBoost** — isotonic calibration, measures probability quality

## Consequences

- Calibration is this project's identity — undersold in most portfolios
- Daily call list ranking depends on probability quality, not just discrimination
- Moderate AUC (0.66) makes calibration especially important
- Report: AUC, F1, Brier score, calibration curve for each model
- No stacking (overkill for 110k rows, 10 features)
