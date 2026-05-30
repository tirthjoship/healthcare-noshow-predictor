# ADR-002: Primary Persona — Outreach / Patient Engagement Team

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Three potential users: front-desk scheduling, outreach/engagement team, operations manager. Each shapes different model output, demo flow, and impact calculation.

## Decision

Primary persona: **outreach / patient engagement team**. They receive a daily call list of high-risk patients, call to confirm/reschedule, and recover appointment slots.

## Consequences

- Streamlit demo anchored as "Today's Call List" with drill-down
- Business impact measured as recovered appointments × revenue
- SHAP narrative framed for non-clinical operations staff
- Secondary consumers (scheduling, ops manager) can use same model output
