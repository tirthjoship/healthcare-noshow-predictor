# ADR-012: SHAP Narrative — Operations Problem, Not Clinical

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Need a framing for SHAP explanations that tells a coherent story. EDA showed clinical features (hypertension, diabetes, handicap) have minimal coefficients. Scheduling features dominate.

## Decision

**Comparison framing**: "The model relies on scheduling features, NOT clinical features. This is an operations problem, not a medical problem."

Global SHAP: bar chart showing lead_time >> age >> SMS >> scholarship >> clinical features near zero.

3 local examples:
1. Young patient, long lead time = high risk
2. Elderly patient, same-day = low risk
3. SMS patient still high risk — confound illustration

## Consequences

- Reframes problem in surprising, insightful way
- Strong interview talking point
- Supports outreach persona (operations staff, not clinicians)
- Global + 3 local examples in `docs/` and Streamlit drill-down
