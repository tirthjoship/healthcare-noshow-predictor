# ADR-009: Fairness Reporting — FPR/FNR + Calibration Per Slice

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Need to report model fairness across demographic groups. Options: AUC per slice, error rates per slice, calibration per slice.

## Decision

**FPR/FNR + calibration per slice**:
- FPR: does model disproportionately flag one group? (wasted calls)
- FNR: does model miss no-shows in a group? (missed outreach)
- Calibration: when model says 30% for group X, is it actually 30%?

Slices: Gender (M/F), Age band (0-18, 19-35, 36-55, 56+), Top 5 neighbourhoods vs rest.

## Consequences

- Ties directly into calibration narrative (ADR-008)
- Operationally meaningful for outreach team fairness
- Output: `reports/fairness.md` — table with N, prevalence, AUC, FPR, FNR, calibration error per slice
- AUC alone would hide interesting failure modes

## Amendment (2026-10-23): what actually shipped

- **Age bands** are 0-17 / 18-34 / 35-54 / 55+ (not 0-18 / 19-35 / 36-55 / 56+).
- **Added slices:** SMS received (makes the ADR-006 confound visible) and welfare enrolment (scholarship).
- **Neighbourhood slice restored:** top 5 by holdout volume vs "other", as originally planned. It had been dropped in the first Phase 2 build.
- **Two operating points:** the single-threshold table (0.22, F1-optimal, **tuned on the June holdout itself, so slightly optimistic**) is kept as a screening view. A **capacity view** at top-20 per day (`capacity_report`) is added and shown first, because it matches the app. See ADR-013 for the age finding.
- Per-slice AUC was not implemented; the capacity view replaced it as the more operational measure.
