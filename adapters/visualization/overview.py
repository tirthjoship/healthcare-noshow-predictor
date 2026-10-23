"""Overview dashboard helpers: pure pandas aggregation, no Streamlit (ADR-010).

The Overview tab reads two exported files (``demo/june_scores.csv`` and
``demo/weekday_lead_rates.csv``). Everything it shows is computed here, so the
numbers, bucketing and shares can be unit tested without a browser.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

AGE_BANDS: tuple[str, ...] = ("0-17", "18-34", "35-54", "55+")
LEAD_BUCKETS: tuple[str, ...] = (
    "Same day",
    "1 to 2 days",
    "3 to 7 days",
    "8 to 14 days",
    "15 to 30 days",
    "31+ days",
)
WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ALL = "All"
OTHER = "Other"
TOP_NEIGHBOURHOODS = 10
# A share gap smaller than this many points is not worth a headline.
MIN_GAP_POINTS = 2.0
# Heatmap cells with fewer appointments than this are greyed out (too noisy).
MIN_CELL = 30


def age_band(age: int) -> str:
    """Age band label; the same rule as ``scripts/run_phase2.py``."""
    if age <= 17:
        return "0-17"
    if age <= 34:
        return "18-34"
    if age <= 54:
        return "35-54"
    return "55+"


def lead_bucket(days: int) -> str:
    """Lead time (calendar days between booking and visit) to a display bucket."""
    if days <= 0:
        return LEAD_BUCKETS[0]
    if days <= 2:
        return LEAD_BUCKETS[1]
    if days <= 7:
        return LEAD_BUCKETS[2]
    if days <= 14:
        return LEAD_BUCKETS[3]
    if days <= 30:
        return LEAD_BUCKETS[4]
    return LEAD_BUCKETS[5]


def top_neighbourhoods(df: pd.DataFrame, n: int = TOP_NEIGHBOURHOODS) -> list[str]:
    """The ``n`` neighbourhoods with the most appointments (ties by name)."""
    counts = df["neighbourhood"].value_counts()
    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [str(name) for name, _ in ranked[:n]]


def apply_filters(
    df: pd.DataFrame,
    day: str,
    band: str = ALL,
    hood: str = ALL,
    top_hoods: list[str] | None = None,
) -> pd.DataFrame:
    """Rows for one appointment day, optionally one age band and neighbourhood.

    ``hood`` is ``ALL``, a name in ``top_hoods`` or ``OTHER`` (every neighbourhood
    outside ``top_hoods``).
    """
    out = df[df["appointment_day"] == day]
    if band != ALL:
        out = out[out["age_band"] == band]
    if hood == OTHER:
        out = out[~out["neighbourhood"].isin(top_hoods or [])]
    elif hood != ALL:
        out = out[out["neighbourhood"] == hood]
    return out


def top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """The ``n`` highest-risk rows, highest first (stable for tied scores)."""
    ranked = df.sort_values("risk_score", ascending=False, kind="stable")
    return ranked.head(max(0, int(n)))


@dataclass(frozen=True)
class Kpis:
    appointments: int
    likely_no_shows: int
    calls_planned: int
    likely_reached: int
    likely_total: float
    reached_total: float


def compute_kpis(df: pd.DataFrame, n: int) -> Kpis:
    """Headline numbers for a filtered board and a call capacity ``n``."""
    called = top_n(df, n)
    likely = float(df["risk_score"].sum())
    reached = float(called["risk_score"].sum())
    return Kpis(
        appointments=len(df),
        likely_no_shows=round(likely),
        calls_planned=len(called),
        likely_reached=round(reached),
        likely_total=likely,
        reached_total=reached,
    )


def band_shares(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per age band: share of likely no-shows vs share of the top-``n`` call list.

    Shares are percentages (0 to 100) and sum to 100 each, or to 0 when empty.
    """
    called = top_n(df, n)
    likely = df.groupby("age_band")["risk_score"].sum().reindex(AGE_BANDS, fill_value=0)
    calls = called.groupby("age_band").size().reindex(AGE_BANDS, fill_value=0)
    likely_total = float(likely.sum())
    calls_total = float(calls.sum())
    return pd.DataFrame(
        {
            "age_band": list(AGE_BANDS),
            "likely_share": [
                100.0 * float(v) / likely_total if likely_total else 0.0 for v in likely
            ],
            "call_share": [
                100.0 * float(v) / calls_total if calls_total else 0.0 for v in calls
            ],
        }
    )


def band_headline(shares: pd.DataFrame) -> str:
    """Action title for the age-band chart, from the largest under-served band."""
    if float(shares["call_share"].sum()) == 0.0:
        return "No calls planned for these filters"
    gap = shares["call_share"] - shares["likely_share"]
    worst = shares.iloc[int(gap.to_numpy().argmin())]
    if float(gap.min()) > -MIN_GAP_POINTS:
        return "Calls track likely no-shows across age bands"
    return (
        f"{worst['age_band']} patients get {worst['call_share']:.0f}% of calls "
        f"for {worst['likely_share']:.0f}% of likely no-shows"
    )


def bucket_rates(rates: pd.DataFrame) -> pd.DataFrame:
    """No-show rate and count per lead bucket, all weekdays pooled, in order."""
    grouped = rates.groupby("lead_bucket")[["appointments", "no_shows"]].sum()
    grouped = grouped.reindex(list(LEAD_BUCKETS)).dropna()
    grouped["no_show_rate"] = grouped["no_shows"] / grouped["appointments"]
    return grouped


def wait_supports_trend(rates: pd.DataFrame) -> bool:
    """True when the pooled rate rises from same-day to the longest wait bucket."""
    pooled = bucket_rates(rates)["no_show_rate"]
    if len(pooled) < 2:
        return False
    return bool(pooled.iloc[-1] > pooled.iloc[0] and pooled.idxmax() != pooled.index[0])


def heatmap_headline(rates: pd.DataFrame) -> str:
    """Action title for the heatmap; only claims a trend the data shows."""
    if wait_supports_trend(rates):
        return "The longer the wait, the more patients miss"
    return "No-show rate by weekday and wait"


def heatmap_caption(rates: pd.DataFrame) -> str:
    """Short caption quoting the shortest and longest wait rates."""
    pooled = bucket_rates(rates)["no_show_rate"]
    if pooled.empty:
        return ""
    return (
        f"{pooled.index[0]}: {100 * pooled.iloc[0]:.1f}% miss. "
        f"{pooled.index[-1]}: {100 * pooled.iloc[-1]:.1f}% miss."
    )


def display_hood(name: str) -> str:
    """Title-case an upper-case neighbourhood name for display."""
    return name.title() if name.isupper() else name
