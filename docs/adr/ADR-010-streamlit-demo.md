# ADR-010: Streamlit Demo — Call List Landing + SHAP Drill-Down

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Reviewers will click the demo first. It needs to tell the story in 30 seconds.

## Decision

Two-page Streamlit app:
1. **Landing: Daily Call List** — top-K patients ranked by no-show risk, configurable K slider, disclaimer banner
2. **Drill-down: Patient Detail** — click a row → SHAP waterfall for that patient, feature values, risk score

## Consequences

- Matches outreach persona (ADR-002) and daily call list intervention (ADR-003)
- User flow: open app → see ranked list → click patient → understand WHY → make call
- Disclaimer banner: "Demo using public benchmark data. Not connected to any health system."
- Max 2 pages — no scope creep into dashboards

## Amendment (2026-10-23): six tabs, plain language first

The app grew beyond two pages because reviewers need context next to the list:
**Start here** (story + headline metrics) → **Daily call list** → **Why this patient?** (SHAP) → **Fairness** (capacity view first, ADR-013 warning, threshold view in an expander) → **Impact** (slots first, editable cost, ADR-004 amendment) → **How we built it** (decision summary, link to `docs/DECISION_TREE.md`).

Labels are plain English (e.g. "Days booked ahead", "Actually missed?"). Technical terms appear only in help text and expanders. The 2-page limit was meant to prevent scope creep, and every added tab serves the "should I trust this list?" question.
