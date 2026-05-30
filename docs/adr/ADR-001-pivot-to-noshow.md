# ADR-001: Pivot from Readmission to Appointment No-Show Prediction

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Original project targeted 30-day hospital readmission prediction using MIMIC-IV. Blocked by credential access. Needed public dataset with clear healthcare ROI and VGH operations narrative fit.

## Decision

Pivot to **medical appointment no-show prediction** using Kaggle Medical Appointments dataset (110,527 rows, 62,299 patients). Keep hexagonal architecture; replace domain models.

## Consequences

- Public dataset, no credential blockers
- Clear business ROI (recoverable appointment revenue)
- Matches VGH outreach/operations narrative
- Domain models renamed: Encounter → Appointment, RiskOutcome → NoShowOutcome
- DataLeakageError retained — "knowable at scheduling time" replaces "pre-discharge"
