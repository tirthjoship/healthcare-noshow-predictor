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

## Amendment (2026-10-23): slots first, per-day precision

- **Lead with recoverable appointment slots, not dollars.** For a public health audience, patient access is the natural unit, and a large dollar headline reads as overselling.
- **Dollars are an editable assumption** in the app (default USD 200, a US primary-care literature estimate), with an "upper bound" note.
- **Bug fix:** precision now comes from the top 20 **per appointment day** (the app's operating point, ADR-005), not the top 20 across the whole holdout. That changes 0.55 to **0.43**, i.e. about **8.7 slots/day, about 191/month** at 20 calls/day.
