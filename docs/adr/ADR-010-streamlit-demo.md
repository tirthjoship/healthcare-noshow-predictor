# ADR-010: Streamlit Demo — Call List Landing + SHAP Drill-Down

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Portfolio reviewers will click the demo. Needs to tell a story in 30 seconds.

## Decision

Two-page Streamlit app:
1. **Landing: Daily Call List** — top-K patients ranked by no-show risk, configurable K slider, disclaimer banner
2. **Drill-down: Patient Detail** — click a row → SHAP waterfall for that patient, feature values, risk score

## Consequences

- Matches outreach persona (ADR-002) and daily call list intervention (ADR-003)
- User flow: open app → see ranked list → click patient → understand WHY → make call
- Disclaimer banner: "Demo using public benchmark data. Not connected to any health system."
- Max 2 pages — no scope creep into dashboards
