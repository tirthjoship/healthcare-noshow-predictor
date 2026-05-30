# ADR-005: Decision Threshold — Capacity-Constrained Top-K

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Model outputs probability. Need to convert to actionable "call or don't call" decision. Options: fixed threshold, top-K, cost-sensitive.

## Decision

**Capacity-constrained top-K**: "team can make N calls/day, give me the N riskiest." Display the implied threshold alongside the list.

## Consequences

- Realistic — outreach teams have fixed bandwidth
- Handles 20% base rate naturally (lots of positives to rank)
- Streamlit shows configurable K slider
- Implied threshold visible for transparency
- Precision@K becomes key evaluation metric
