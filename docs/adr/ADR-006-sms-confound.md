# ADR-006: SMS Confound Handling — Include with Documentation

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

SMS_received=1 correlates with higher no-show rate (Simpson's paradox). SMS is sent as intervention to patients with long lead times (high-risk). Not a causal predictor.

## Decision

**Include SMS as feature** with prominent documentation of the confound:
- Model card: SMS coefficient is not causal
- SHAP narrative: explain Simpson's paradox
- README Limitations section
- Interview talking point

## Consequences

- Preserves useful signal (SMS proxies "clinic flagged as risky")
- Demonstrates understanding of observational data limitations
- Documented in 3+ locations to prevent misinterpretation
- Strong portfolio differentiator for stats-savvy interviewers
