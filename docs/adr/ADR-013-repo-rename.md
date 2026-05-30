# ADR-013: Repo Rename to healthcare-noshow-predictor

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Repo was `patient-readmission-risk-engine`. After pivot to no-show prediction, name no longer matches. Need portfolio consistency with `supply-chain-optimization-ml` and `multi-modal-stock-recommender`.

## Decision

Rename to **`healthcare-noshow-predictor`**.
- Pattern: `[domain]-[problem]-[suffix]`
- GitHub remote: `https://github.com/tirthjoship/healthcare-noshow-predictor.git`

## Consequences

- Folder rename deferred until user confirms local path change
- GitHub remote URL updated
- README, CONTEXT.md, all docs updated to new name
- Old remote references removed
