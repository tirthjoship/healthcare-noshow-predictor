# ADR-007: Neighbourhood Encoding — Target Encoding (Train-Only)

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

81 unique neighbourhoods. High cardinality categorical. EDA shows no-show rates range 15-26% across neighbourhoods — real signal.

## Decision

**Target encoding**, fit on training data only, with fold-aware implementation during CV. Use sklearn `TargetEncoder` or manual fold-aware implementation.

## Consequences

- Captures neighbourhood-level signal without 81 dummy columns
- Leakage prevention: encoder never sees test/validation labels
- Consistent with DataLeakageError domain theme
- Must be wrapped in pipeline or applied carefully in CV loop
