"""Unit tests for the Overview aggregation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from adapters.visualization import overview as ov


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "appointment_day": ["d1"] * 5 + ["d2"],
            "age": [5, 20, 40, 70, 80, 30],
            "age_band": ["0-17", "18-34", "35-54", "55+", "55+", "18-34"],
            "neighbourhood": ["A", "A", "B", "B", "C", "A"],
            "lead_time_days": [0, 1, 5, 20, 40, 3],
            "risk_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
            "actual_no_show": [0, 0, 1, 0, 1, 1],
        }
    )


@pytest.mark.parametrize(
    "age,band",
    [
        (0, "0-17"),
        (17, "0-17"),
        (18, "18-34"),
        (34, "18-34"),
        (35, "35-54"),
        (54, "35-54"),
        (55, "55+"),
        (99, "55+"),
    ],
)
def test_age_band(age: int, band: str) -> None:
    assert ov.age_band(age) == band


@pytest.mark.parametrize(
    "days,bucket",
    [
        (0, "Same day"),
        (1, "1 to 2 days"),
        (2, "1 to 2 days"),
        (3, "3 to 7 days"),
        (7, "3 to 7 days"),
        (8, "8 to 14 days"),
        (14, "8 to 14 days"),
        (15, "15 to 30 days"),
        (30, "15 to 30 days"),
        (31, "31+ days"),
        (179, "31+ days"),
    ],
)
def test_lead_bucket(days: int, bucket: str) -> None:
    assert ov.lead_bucket(days) == bucket


def test_filters_and_other_bucket() -> None:
    df = _df()
    assert len(ov.apply_filters(df, "d1")) == 5
    assert len(ov.apply_filters(df, "d1", band="55+")) == 2
    assert len(ov.apply_filters(df, "d1", hood="B")) == 2
    top = ["A", "B"]
    other = ov.apply_filters(df, "d1", hood=ov.OTHER, top_hoods=top)
    assert list(other["neighbourhood"]) == ["C"]


def test_top_neighbourhoods_by_volume_then_name() -> None:
    assert ov.top_neighbourhoods(_df(), 2) == ["A", "B"]


def test_kpis() -> None:
    day = ov.apply_filters(_df(), "d1")
    k = ov.compute_kpis(day, 2)
    assert k.appointments == 5
    assert k.likely_no_shows == 2  # round(1.5) is 2
    assert k.calls_planned == 2
    assert k.likely_total == pytest.approx(1.5)
    assert k.reached_total == pytest.approx(0.9)
    assert k.likely_reached == 1


def test_calls_planned_capped_by_appointments() -> None:
    k = ov.compute_kpis(ov.apply_filters(_df(), "d2"), 100)
    assert (k.appointments, k.calls_planned) == (1, 1)


def test_empty_board() -> None:
    k = ov.compute_kpis(ov.apply_filters(_df(), "d1", band="0-17", hood="B"), 20)
    assert (k.appointments, k.calls_planned, k.likely_no_shows) == (0, 0, 0)
    shares = ov.band_shares(_df().iloc[0:0], 5)
    assert shares["likely_share"].sum() == 0
    assert ov.band_headline(shares) == "No calls planned for these filters"


def test_top_n_order_and_clamp() -> None:
    day = ov.apply_filters(_df(), "d1")
    assert list(ov.top_n(day, 3)["risk_score"]) == [0.5, 0.4, 0.3]
    assert ov.top_n(day, -1).empty


def test_band_shares_sum_and_values() -> None:
    day = ov.apply_filters(_df(), "d1")
    s = ov.band_shares(day, 2)
    assert s["likely_share"].sum() == pytest.approx(100)
    assert s["call_share"].sum() == pytest.approx(100)
    row = s[s["age_band"] == "55+"].iloc[0]
    assert row["likely_share"] == pytest.approx(100 * 0.9 / 1.5)
    assert row["call_share"] == pytest.approx(100)


def test_band_headline_gap() -> None:
    day = ov.apply_filters(_df(), "d1")
    assert ov.band_headline(ov.band_shares(day, 2)) != ""
    even = pd.DataFrame(
        {
            "age_band": list(ov.AGE_BANDS),
            "likely_share": [25.0] * 4,
            "call_share": [25.0] * 4,
        }
    )
    assert "track" in ov.band_headline(even)
    skew = even.assign(
        call_share=[40.0, 40.0, 19.0, 1.0], likely_share=[10.0, 30.0, 40.0, 20.0]
    )
    assert ov.band_headline(skew).startswith("35-54 patients get 19% of calls")


def _rates(vals: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weekday": ["Mon"] * len(vals),
            "lead_bucket": list(ov.LEAD_BUCKETS)[: len(vals)],
            "appointments": [100] * len(vals),
            "no_shows": [int(v * 100) for v in vals],
        }
    )


def test_heatmap_headline_only_when_supported() -> None:
    up = _rates([0.05, 0.2, 0.25, 0.3, 0.33, 0.34])
    down = _rates([0.4, 0.3, 0.25, 0.2, 0.15, 0.1])
    assert ov.heatmap_headline(up) == "The longer the wait, the more patients miss"
    assert ov.heatmap_headline(down) == "No-show rate by weekday and wait"
    assert "5.0% miss" in ov.heatmap_caption(up)
    assert ov.display_hood("JARDIM CAMBURI") == "Jardim Camburi"
