# ADR-013: Older Patients Are Left Off a Raw-Risk Call List (Disclose, Don't Patch Yet)

**Date:** 2026-10-23
**Status:** Accepted
**Deciders:** Tirth Joshi

## Context

The Phase 2 fairness report (ADR-009) measured errors at a single probability threshold (0.22, F1-optimal). That threshold was tuned on the June holdout itself, so it is slightly optimistic, and it flags about half of all patients. The app does not work that way: it calls the **top K per day** (ADR-005). We added a capacity view (`capacity_report`, K = 20 per day) to measure fairness at the setting actually used.

**Finding (June 2016 holdout, after the ADR-014 fix):**

| Age band | Share of patients | Share of no-shows | Share of the call list | FNR at threshold 0.22 |
|---|---|---|---|---|
| 0-17 | 24.6% | 26.8% | 36.7% | 15.5% |
| 18-34 | 21.1% | 25.5% | 46.7% | 6.6% |
| 35-54 | 27.3% | 26.5% | 15.8% | 17.9% |
| **55+** | 27.0% | 21.2% | **0.8%** (1 of 120 calls) | **59.3%** |

Calibration is fine in every age band (ECE 0.016 to 0.023), so the scores are honest. The cause is the ranking rule. Patients 55+ have a lower no-show rate (14.4% vs 22.3% for 18-34), so their individual risks rarely reach the top of a short list. One threshold or one global ranking for everyone hands almost the whole list to younger patients.

**Plain English:** the model isn't wrong about older patients. It's correct that each one is less likely to miss. But when you only make 20 calls, "call the riskiest" means older patients almost never get a call, even though they make up a fifth of the missed appointments.

## Decision

1. **Disclose prominently** in the app (Fairness tab warning plus a Start-here summary), in the README and in the decision tree.
2. **No change to the model or ranking now.** Any fix is a policy choice, not a modelling bug:
   - per-group capacity (reserve some calls per age band),
   - per-group thresholds (equal FNR), or
   - ranking by expected benefit (risk × chance a call helps), which needs intervention data we don't have.
   Each option trades total hits for equity, and that trade-off needs clinical and equity input.
3. **Future work (v2):** evaluate per-group capacity on the holdout and report the hit-rate cost of equalising capture.

## Consequences

- The public demo shows a real limitation instead of hiding it. That's the main talking point for health-sector reviewers.
- The ADR-009 threshold table stays, labeled as a screening view tuned on the test month.
- Small-sample caveat: the capacity view covers 6 June days (120 calls). The direction is clear; the exact percentages are noisy.
