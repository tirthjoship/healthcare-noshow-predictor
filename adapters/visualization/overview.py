"""Overview dashboard helpers: pure pandas aggregation, no Streamlit (ADR-010).

The Overview tab reads two exported files (``demo/june_scores.csv.gz`` and
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


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add the age band derived from ``age`` (kept out of the exported file)."""
    out = df.copy()
    bands = pd.cut(out["age"], [-1, 17, 34, 54, 10_000], labels=list(AGE_BANDS))
    out["age_band"] = bands.astype(str)
    return out


def calibration_gap_pct(df: pd.DataFrame) -> float:
    """How far the summed risk runs above (positive) the actual no-show count, in %."""
    actual = float(df["actual_no_show"].sum())
    if actual == 0:
        return 0.0
    return 100.0 * (float(df["risk_score"].sum()) / actual - 1.0)


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
    """The ``n`` highest-risk rows, highest first (``day_rank`` breaks ties)."""
    if "day_rank" in df.columns:
        # Rank from the unrounded score, exported per day; order is preserved
        # inside any filtered subset of a day.
        ranked = df.sort_values("day_rank", kind="stable")
    else:
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
    random_reached: float


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
        random_reached=len(called) * likely / len(df) if len(df) else 0.0,
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


def band_headline(shares: pd.DataFrame, calls: int = 0) -> str:
    """Action title for the age-band chart, from the band with the biggest gap."""
    if float(shares["call_share"].sum()) == 0.0:
        return "No calls planned for these filters"
    gap = shares["call_share"] - shares["likely_share"]
    if float(gap.min()) > -MIN_GAP_POINTS:
        return "Calls track likely no-shows across age bands"
    worst = shares.iloc[int(gap.to_numpy().argmin())]
    likely = f"{worst['likely_share']:.0f}% of likely no-shows"
    if float(worst["call_share"]) == 0.0:
        label = f"top {calls}" if calls else "call list"
        return (
            f"No {worst['age_band']} patients make the {label}, "
            f"though they account for {likely}"
        )
    return (
        f"{worst['age_band']} patients get {worst['call_share']:.0f}% of calls "
        f"but account for {likely}"
    )


def bucket_rates(rates: pd.DataFrame) -> pd.DataFrame:
    """No-show rate and count per lead bucket, all weekdays pooled, in order."""
    grouped = rates.groupby("lead_bucket")[["appointments", "no_shows"]].sum()
    grouped = grouped.reindex(list(LEAD_BUCKETS)).dropna()
    grouped["no_show_rate"] = grouped["no_shows"] / grouped["appointments"]
    return grouped


def visible_weekdays(rates: pd.DataFrame) -> list[str]:
    """Weekdays (Mon first) with at least one cell of ``MIN_CELL`` appointments."""
    big = rates[rates["appointments"] >= MIN_CELL]["weekday"]
    return [d for d in WEEKDAYS if d in set(big)]


def heatmap_data(rates: pd.DataFrame) -> pd.DataFrame:
    """Heatmap rows without the weekdays that have no usable cell."""
    return rates[rates["weekday"].isin(visible_weekdays(rates))].copy()


def wait_supports_trend(rates: pd.DataFrame) -> bool:
    """True when every wait bucket misses more often than same-day bookings."""
    pooled = bucket_rates(rates)["no_show_rate"]
    if len(pooled) < 2 or pooled.index[0] != LEAD_BUCKETS[0]:
        return False
    return bool((pooled.iloc[1:] > pooled.iloc[0]).all())


def heatmap_headline(rates: pd.DataFrame) -> str:
    """Action title for the heatmap; only claims a trend the data shows."""
    if not wait_supports_trend(rates):
        return "No-show rate by weekday and wait"
    pooled = bucket_rates(rates)["no_show_rate"] * 100
    return (
        f"Same-day bookings rarely miss ({pooled.iloc[0]:.0f}%); any wait raises "
        f"misses to {pooled.iloc[1:].min():.0f} to {pooled.iloc[1:].max():.0f}%"
    )


def heatmap_caption(rates: pd.DataFrame) -> str:
    """Caption saying what population the heatmap covers."""
    total = int(rates["appointments"].sum())
    return (
        f"All {total // 1000}K appointments, Apr to Jun 2016, not the "
        f"selected day. Grey: under {MIN_CELL} appointments."
    )


def bullet_headline(k: Kpis) -> str:
    """Action title for the coverage chart."""
    if k.calls_planned == 0:
        return "No calls planned for these filters"
    text = f"{k.reached_total:.0f} of {k.calls_planned} calls reach a likely no-show"
    if k.random_reached > 0:
        text += f", about {k.reached_total / k.random_reached:.1f}x random"
    return text


def display_hood(name: str) -> str:
    """Title-case an upper-case neighbourhood name for display."""
    return name.title() if name.isupper() else name
