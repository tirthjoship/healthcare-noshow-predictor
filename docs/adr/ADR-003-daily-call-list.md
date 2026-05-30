# ADR-003: Intervention Model — Daily Call List

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Model outputs probability. Need to define how outreach team consumes it: triggered at booking, daily batch, or weekly batch.

## Decision

**Daily call list** generated each morning. Top-K highest-risk patients with appointments in next 3-7 days. Team calls during morning huddle.

## Consequences

- Natural fit for lead_time signal (patients enter high-risk window as appointment approaches)
- Capacity-constrained top-K ranking (ADR-005) determines list size
- Streamlit demo shows daily list as landing page
- No need for SMS API integration (out of scope)
