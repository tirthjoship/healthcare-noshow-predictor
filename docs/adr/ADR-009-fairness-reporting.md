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
