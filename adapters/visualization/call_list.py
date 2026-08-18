"""Call-list ranking — pure helpers for the outreach demo (ADR-003, ADR-005).

The outreach team states a daily call capacity K; the system returns the K
highest-risk patients. These helpers are UI-agnostic (list[dict] in, list[dict]
out) so they can be unit-tested without Streamlit.
"""

from __future__ import annotations


def rank_call_list(
    candidates: list[dict[str, object]],
    capacity_k: int,
    score_key: str = "risk_score",
) -> list[dict[str, object]]:
    """Return the top-K candidates by risk score, highest first.

    Args:
        candidates: Scored appointment records (each with ``score_key``).
        capacity_k: Daily call capacity K (clamped to >= 0).
        score_key: Record key holding the risk score.

    Returns:
        The K highest-risk records, sorted descending by score.
    """
    ranked = sorted(
        candidates, key=lambda c: float(c[score_key]), reverse=True  # type: ignore[arg-type]
    )
    return ranked[: max(0, capacity_k)]


def precision_at_capacity(
    ranked: list[dict[str, object]],
    label_key: str = "actual_no_show",
) -> float:
    """Fraction of the ranked list that were true no-shows.

    Args:
        ranked: The (already ranked, already truncated) call list.
        label_key: Record key holding the ground-truth no-show flag (0/1/bool).

    Returns:
        Precision in [0, 1], or 0.0 for an empty list.
    """
    if not ranked:
        return 0.0
    hits = sum(1 for r in ranked if bool(r[label_key]))
    return hits / len(ranked)
