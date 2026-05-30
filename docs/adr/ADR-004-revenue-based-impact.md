# ADR-004: Business Impact Metric — Revenue-Based

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

Need to quantify value of recovered appointments. Options: revenue-based, cost-based, or hybrid.

## Decision

**Revenue-based** with explicit assumptions labeled:
```
cost_per_noshow = $200 (primary care literature estimate)
monthly_value = top_decile_recovered × precision_at_threshold × $200
```

## Consequences

- Simple, defensible, one number with citation
- All assumptions clearly labeled in `docs/BUSINESS_IMPACT.md`
- Conservative enough for portfolio credibility
- Reviewer can substitute their own cost_per_noshow easily
