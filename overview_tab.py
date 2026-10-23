"""Overview tab: a one-screen BI style dashboard (filters, KPIs, four cards)."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

import ui_theme as ui
from adapters.visualization import overview as ov

ROOT = Path(__file__).resolve().parent
SCORES_CSV = ROOT / "demo" / "june_scores.csv.gz"
RATES_CSV = ROOT / "demo" / "weekday_lead_rates.csv"
HEAT_H = 330
BAND_H = 150
LIST_H = 412
SERIES = ["Likely no-shows", "Call list"]


@st.cache_data  # type: ignore[untyped-decorator]
def load_scores() -> pd.DataFrame:
    return ov.enrich(pd.read_csv(SCORES_CSV))


@st.cache_data  # type: ignore[untyped-decorator]
def load_rates() -> pd.DataFrame:
    return pd.read_csv(RATES_CSV)


def fmt_day(iso: str) -> str:
    ts = pd.Timestamp(iso)
    return f"{ts:%a} {ts.day} {ts:%b %Y}"


def _on_pick(chart_key: str) -> None:
    """Chart click -> age band filter (clicking the active band clears it)."""
    event = st.session_state.get(chart_key)
    picked: list[dict[str, str]] = []
    if event is not None:
        picked = event["selection"].get("band_pick", [])
    current = st.session_state.get("ov_band", ov.ALL)
    if not picked:
        return
    band = picked[0].get("age_band", ov.ALL)
    st.session_state["ov_band"] = ov.ALL if band == current else band


def _fit(chart: Any, height: int) -> Any:
    """Apply the shared theme and keep axis labels inside the chart frame."""
    sized = chart.properties(
        height=height,
        padding={"left": 12, "right": 8, "top": 4, "bottom": 2},
        autosize=alt.AutoSizeParams(type="fit", contains="padding"),
    )
    return ui.style_chart(sized)


def _heatmap(rates: pd.DataFrame) -> Any:
    data = ov.heatmap_data(rates)
    data["rate_pct"] = data["no_show_rate"] * 100
    data["rate_label"] = data["rate_pct"].map(lambda v: f"{v:.0f}%")
    data["enough"] = data["appointments"] >= ov.MIN_CELL
    days = ov.visible_weekdays(rates)
    x = alt.X(
        "lead_bucket:N",
        sort=list(ov.LEAD_BUCKETS),
        title="Days between booking and visit",
        axis=alt.Axis(
            labelAngle=0,
            labelFontSize=10,
            labelExpr="replace(datum.label, ' days', '')",
        ),
    )
    y = alt.Y("weekday:N", sort=days, title=None)
    tip = [
        alt.Tooltip("weekday:N", title="Weekday"),
        alt.Tooltip("lead_bucket:N", title="Wait"),
        alt.Tooltip("rate_pct:Q", title="No-show rate (%)", format=".1f"),
        alt.Tooltip("appointments:Q", title="Appointments", format=","),
    ]
    good = data[data["enough"]]
    sparse = data[~data["enough"]]
    scale = alt.Scale(
        domain=[0, 15, 28, 40],
        range=["#EEF5F8", ui.ACCENT_SOFT, ui.ACCENT, "#0F3F52"],
        clamp=True,
    )
    legend = alt.Legend(
        title="No-show rate (%)",
        orient="top",
        direction="horizontal",
        gradientLength=140,
        gradientThickness=8,
        titleFontSize=10,
        labelFontSize=10,
        values=[0, 10, 20, 30, 40],
    )
    base_good = alt.Chart(good).encode(x=x, y=y, tooltip=tip)
    cells = base_good.mark_rect(stroke="#fff", strokeWidth=2).encode(
        color=alt.Color("rate_pct:Q", scale=scale, legend=legend)
    )
    labels = base_good.mark_text(fontSize=11, font="IBM Plex Mono").encode(
        text="rate_label:N",
        color=alt.condition(
            alt.datum.rate_pct > 24, alt.value("#FFFFFF"), alt.value(ui.INK)
        ),
    )
    grey = (
        alt.Chart(sparse)
        .mark_rect(stroke="#fff", strokeWidth=2, color="#F2F4F6")
        .encode(x=x, y=y, tooltip=tip)
    )
    return _fit(alt.layer(grey, cells, labels), HEAT_H)


def _butterfly(shares: pd.DataFrame, band: str) -> Any:
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "age_band": shares["age_band"],
                    "series": SERIES[0],
                    "share": shares["likely_share"],
                    "signed": -shares["likely_share"],
                }
            ),
            pd.DataFrame(
                {
                    "age_band": shares["age_band"],
                    "series": SERIES[1],
                    "share": shares["call_share"],
                    "signed": shares["call_share"],
                }
            ),
        ]
    )
    long["opacity"] = [1.0 if band in (ov.ALL, b) else 0.35 for b in long["age_band"]]
    pick = alt.selection_point(name="band_pick", fields=["age_band"])
    span = max(10.0, float(long["share"].max()) * 1.25)
    bars = (
        alt.Chart(long)
        .mark_bar(cornerRadius=2)
        .encode(
            y=alt.Y(
                "age_band:N",
                sort=list(ov.AGE_BANDS),
                title=None,
                axis=alt.Axis(labelPadding=6),
            ),
            x=alt.X(
                "signed:Q",
                title="Share of total (%)",
                scale=alt.Scale(domain=[-span, span]),
                axis=alt.Axis(labelExpr="abs(datum.value) + '%'", tickCount=5),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=SERIES, range=[ui.ACCENT_SOFT, ui.ACCENT]),
                legend=alt.Legend(title=None, columnPadding=18, symbolSize=90),
            ),
            opacity=alt.Opacity("opacity:Q", scale=None),
            tooltip=[
                alt.Tooltip("age_band:N", title="Age band"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("share:Q", title="Share (%)", format=".1f"),
            ],
        )
        .add_params(pick)
    )
    return _fit(bars, BAND_H)


def _bullet(k: ov.Kpis) -> Any:
    """Three bars on one 0 to N calls scale: plan, reached, and random calling."""
    data = pd.DataFrame(
        {
            "measure": [
                "Calls planned",
                "Misses reached",
                "At random",
            ],
            "value": [float(k.calls_planned), k.reached_total, k.random_reached],
            "color": ["#DCE9EF", ui.ACCENT, ui.ACCENT_SOFT],
        }
    )
    data["label"] = data["value"].map(lambda v: f"{v:.1f}")
    order = list(data["measure"])
    top = max(float(k.calls_planned), 1.0) * 1.12
    base = alt.Chart(data).encode(
        y=alt.Y("measure:N", sort=order, title=None, axis=alt.Axis(labelLimit=170)),
        x=alt.X(
            "value:Q",
            scale=alt.Scale(domain=[0, top]),
            title="Calls and expected no-shows",
            axis=alt.Axis(tickMinStep=1, tickCount=5),
        ),
        tooltip=[
            alt.Tooltip("measure:N", title="Measure"),
            alt.Tooltip("value:Q", title="Patients", format=".1f"),
        ],
    )
    bars = base.mark_bar(size=22).encode(color=alt.Color("color:N", scale=None))
    text = base.mark_text(align="left", dx=5, fontSize=12).encode(text="label:N")
    return _fit(alt.layer(bars, text), BAND_H)


def render() -> None:
    scores = load_scores()
    rates = load_rates()
    days = sorted(scores["appointment_day"].unique().tolist())
    hoods = ov.top_neighbourhoods(scores)

    with st.container(key="ov_rail"):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 1.4], gap="medium")
        day = c1.selectbox("Appointment day", days, key="ov_day", format_func=fmt_day)
        band = c2.selectbox("Age band", [ov.ALL, *ov.AGE_BANDS], key="ov_band")
        hood = c3.selectbox(
            "Neighbourhood",
            [ov.ALL, *hoods, ov.OTHER],
            key="ov_hood",
            format_func=lambda h: h if h in (ov.ALL, ov.OTHER) else ov.display_hood(h),
        )
        n = c4.slider("Calls per day", 5, 100, 20, key="ov_calls")

    board = ov.apply_filters(scores, day, band, hood, hoods)
    k = ov.compute_kpis(board, n)
    gap = ov.calibration_gap_pct(scores)
    st.markdown(
        '<div class="ov-kpis">'
        + ui.tiles(
            [
                ui.tile(
                    f"{k.appointments:,}",
                    "Appointments",
                    "neutral",
                    "Booked appointments on the selected day that match the filters.",
                ),
                ui.tile(
                    f"{k.likely_no_shows:,}",
                    "Likely no-shows",
                    "caution",
                    "The model's expected count: the sum of every patient's risk "
                    f"score. On June test days it ran about {gap:.0f}% above the "
                    "actual number of no-shows.",
                ),
                ui.tile(
                    f"{k.calls_planned:,}",
                    "Calls planned",
                    "accent",
                    "The calls per day you chose, or fewer if fewer patients match.",
                ),
                ui.tile(
                    f"{k.likely_reached:,}",
                    "Likely misses reached",
                    "good",
                    "The expected number of no-shows among the highest-risk patients "
                    "who would get a call.",
                ),
            ],
            columns=4,
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card("ov_heat"):
            st.markdown(
                ui.card_title(ov.heatmap_headline(rates)), unsafe_allow_html=True
            )
            st.altair_chart(_heatmap(rates), width="stretch", theme=None)
            st.markdown(
                f'<div class="ov-cap">{ui.esc(ov.heatmap_caption(rates))}</div>',
                unsafe_allow_html=True,
            )
    with right:
        with ui.card("ov_list"):
            st.markdown(
                ui.card_title(f"Call these {k.calls_planned} first"),
                unsafe_allow_html=True,
            )
            called = ov.top_n(board, n)
            if called.empty:
                st.caption("No appointments match these filters.")
            else:
                table = pd.DataFrame(
                    {
                        "Rank": range(1, len(called) + 1),
                        "Age": called["age"].to_numpy(),
                        "Neighbourhood": [
                            ov.display_hood(h) for h in called["neighbourhood"]
                        ],
                        "Lead days": called["lead_time_days"].to_numpy(),
                        "Risk": called["risk_score"].to_numpy(),
                    }
                )
                st.dataframe(
                    table,
                    hide_index=True,
                    height=LIST_H,
                    width="stretch",
                    column_config={
                        "Rank": st.column_config.NumberColumn(width=44),
                        "Age": st.column_config.NumberColumn(width=44),
                        "Neighbourhood": st.column_config.TextColumn(width=128),
                        "Lead days": st.column_config.NumberColumn(width=70),
                        "Risk": st.column_config.ProgressColumn(
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                            help="Calibrated chance of a no-show.",
                        ),
                    },
                )

    left2, right2 = st.columns(2, gap="medium")
    with left2:
        with ui.card("ov_band"):
            base = ov.apply_filters(scores, day, ov.ALL, hood, hoods)
            shares = ov.band_shares(base, n)
            st.markdown(
                ui.card_title(ov.band_headline(shares, k.calls_planned)),
                unsafe_allow_html=True,
            )
            key = f"ov_band_chart_{band}"
            st.altair_chart(
                _butterfly(shares, band),
                width="stretch",
                theme=None,
                key=key,
                on_select=partial(_on_pick, key),
                selection_mode="band_pick",
            )
            band_cap = (
                "Still compares all age bands for the selected day. Click the "
                "band again to clear the filter."
                if band != ov.ALL
                else "Click an age band to filter the board."
            )
            st.markdown(
                f'<div class="ov-cap">{ui.esc(band_cap)}</div>',
                unsafe_allow_html=True,
            )
    with right2:
        with ui.card("ov_bullet"):
            st.markdown(
                ui.card_title(ov.bullet_headline(k)),
                unsafe_allow_html=True,
            )
            st.altair_chart(_bullet(k), width="stretch", theme=None)
            st.markdown(
                '<div class="ov-cap">Random calling is the same number of calls '
                "at the average risk of the filtered patients.</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="ov-foot"><span>Public Kaggle data, June 2016 test days. '
        "Not connected to any health system.</span>"
        "<span>Read the full brief in the Brief tab.</span></div>",
        unsafe_allow_html=True,
    )
