"""Visual layer for the Streamlit app: a "layered decision brief".

Every tab has three layers: a decision layer (the bottom line), an evidence layer
(charts in bordered cards, each with a one-line takeaway) and a method layer
(tooltips plus a collapsed "How we know" expander).

The helpers that build markup are pure functions returning strings so they can be
unit tested without Streamlit. Only ``inject_css`` and ``card`` touch Streamlit.
"""

from __future__ import annotations

import html
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

# --------------------------------------------------------------------------- palette
PAPER = "#FFFFFF"
INK = "#0D0F12"
SLATE = "#5C6B7A"
CARD_BORDER = "#EAECEF"
ACCENT = "#1F6F8B"
ACCENT_SOFT = "#9CC3D5"
GOOD = "#2F7D4F"
CAUTION = "#B7791F"
PROBLEM = "#B3261E"

FONT_HEAD = "Space Grotesk, system-ui, sans-serif"
FONT_BODY = "Inter, system-ui, sans-serif"
FONT_MONO = "IBM Plex Mono, ui-monospace, monospace"

TONE_COLOR = {
    "good": GOOD,
    "caution": CAUTION,
    "problem": PROBLEM,
    "neutral": INK,
    "accent": ACCENT,
}
TONE_TINT = {
    "good": "#E3F3E9",
    "caution": "#F8EDD6",
    "problem": "#F9E2E0",
    "neutral": "#F2F4F6",
    "accent": "#E4F0F5",
}


def normalize_tone(tone: str) -> str:
    """Map any tone string to a known tone; unknown values become neutral."""
    return tone if tone in TONE_COLOR else "neutral"


def tone_color(tone: str) -> str:
    return TONE_COLOR[normalize_tone(tone)]


def esc(text: object) -> str:
    """Escape text for safe insertion into HTML rendered through Markdown.

    Besides HTML special characters, a dollar sign becomes an entity so that
    Markdown does not treat a pair of them as a math block.
    """
    return html.escape(str(text), quote=True).replace("$", "&#36;")


# --------------------------------------------------------------------------- CSS
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {{
  --ink: {INK}; --slate: {SLATE}; --line: {CARD_BORDER}; --accent: {ACCENT};
  --good: {GOOD}; --caution: {CAUTION}; --problem: {PROBLEM};
  --head: 'Space Grotesk', system-ui, sans-serif;
  --body: 'Inter', system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
}}
.stApp {{ background: {PAPER}; }}
.stApp, .stApp p, .stApp li, .stApp label {{ font-family: var(--body); color: var(--ink); }}
.block-container {{ padding-top: 2.6rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ font-family: var(--head) !important; letter-spacing: -0.02em; color: var(--ink); }}
h3 {{ font-size: 1.15rem !important; }}

/* tabs */
[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--line); }}
[data-baseweb="tab"] {{ font-family: var(--mono) !important; font-size: 12.5px; letter-spacing: .04em; }}
[data-baseweb="tab"] p {{ font-family: var(--mono) !important; font-size: 12.5px; }}
[data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}

/* header */
.nb-head {{ margin: 0 0 18px; }}
.nb-eyebrow {{ white-space: nowrap; font: 600 11.5px var(--mono); letter-spacing: .16em; color: var(--accent); text-transform: uppercase; }}
.nb-q {{ font: 700 38px/1.1 var(--head); letter-spacing: -0.025em; margin: 8px 0 8px; color: var(--ink); }}
.nb-sum {{ font-size: 15px; color: var(--slate); max-width: 780px; margin: 0 0 12px; line-height: 1.5; }}
.nb-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.nb-chip {{ font: 500 12px var(--mono); background: #F6F7F9; border: 1px solid var(--line);
  border-radius: 20px; padding: 4px 11px; color: #48505A; position: relative; display: inline-block; }}
.nb-chip.good {{ background: {TONE_TINT['good']}; border-color: #BFE3CD; color: var(--good); }}
.nb-chip.caution {{ background: {TONE_TINT['caution']}; border-color: #EBD6A4; color: var(--caution); }}
.nb-chip.problem {{ background: {TONE_TINT['problem']}; border-color: #EDBDB9; color: var(--problem); }}
.nb-chip.accent {{ background: {TONE_TINT['accent']}; border-color: #BCD8E3; color: var(--accent); }}

/* info icon with hover tooltip */
.ci {{ display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px;
  border-radius: 50%; border: 1px solid #C2C7CD; color: #8B919A; font: 600 10px var(--body);
  cursor: help; position: relative; margin-left: 6px; vertical-align: middle; }}
.ci:hover {{ border-color: var(--accent); color: var(--accent); }}
.ci:hover::after {{ content: attr(data-def); position: absolute; left: 0; top: 1.8em; z-index: 50;
  background: #111; color: #fff; font: 400 12px/1.5 var(--body); width: 280px; padding: 10px 12px;
  border-radius: 9px; white-space: normal; box-shadow: 0 8px 24px rgba(0,0,0,.3); text-align: left; }}
.ci.r:hover::after, .tile:nth-child(n+3) .ci:hover::after {{ left: auto; right: 0; }}

/* bottom line */
.bl {{ border: 1px solid var(--line); border-left: 4px solid var(--accent);
  background: linear-gradient(180deg, #F5FAFC, #FFFFFF); border-radius: 12px;
  padding: 18px 22px 16px; margin: 6px 0 18px; }}
.bl .lab {{ font: 600 11px var(--mono); letter-spacing: .16em; text-transform: uppercase; color: var(--accent); }}
.bl .q {{ font: 600 19px/1.3 var(--head); margin: 6px 0 10px; color: var(--ink); }}
.bl .verd {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }}
.bl .ans {{ display: inline-block; padding: 5px 14px; border-radius: 20px; font: 600 14.5px var(--body); }}
.bl p {{ margin: 0 0 6px; font-size: 14.5px; line-height: 1.55; color: #2E343B; max-width: 860px; }}
.bl .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px 28px;
  border-top: 1px solid var(--line); margin-top: 12px; padding-top: 11px; }}
.bl .meta .k {{ font: 600 10.5px var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--slate); }}
.bl .meta .v {{ font-size: 13.5px; color: var(--ink); margin-top: 2px; line-height: 1.45; }}

/* number tiles */
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 4px 0 16px; }}
.tile {{ border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 12px; background: #fff; }}
.tile .num {{ font: 700 34px/1.05 var(--head); letter-spacing: -0.02em; }}
.tile .lab {{ font-size: 13px; color: var(--slate); margin-top: 6px; line-height: 1.35; }}
.tile .sub {{ font: 500 11.5px var(--mono); color: var(--slate); margin-top: 6px; }}

/* chart cards */
[class*="st-key-card_"] {{ border: 1px solid var(--line) !important; border-radius: 12px !important;
  background: #fff; padding: 6px 8px 4px; overflow: visible; }}
.cardtitle {{ font: 600 14px var(--body); color: var(--ink); margin: 2px 0 2px; }}
.takeaway {{ font-size: 13.5px; color: #2E343B; line-height: 1.5; margin: 2px 2px 4px; padding-left: 10px;
  border-left: 3px solid {ACCENT_SOFT}; }}
.layer {{ font: 600 10.5px var(--mono); letter-spacing: .16em; text-transform: uppercase; color: var(--slate);
  margin: 22px 0 8px; }}
.note {{ border: 1px solid #EBD6A4; background: #FCF7EA; border-radius: 10px; padding: 12px 16px;
  font-size: 13.5px; line-height: 1.55; color: #3A3320; margin: 10px 0; }}
.note.accent {{ border-color: #BCD8E3; background: #EEF6F9; color: #1B3A47; }}
.tile.bare {{ border: none; padding: 4px 0 0; }}
.note b {{ color: var(--ink); }}
[data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 12px; }}
[data-testid="stExpander"] summary p {{ font-family: var(--mono); font-size: 12.5px; letter-spacing: .04em; }}
/* overview dashboard */
.ov-kpis .tiles {{ margin: 2px 0 10px; gap: 10px; }}
.ov-kpis .tile {{ padding: 9px 14px 8px; }}
.ov-kpis .tile .num {{ font-size: 28px; }}
.ov-kpis .tile .lab {{ margin-top: 3px; font-size: 12.5px; }}
[class*="st-key-card_ov_"] {{ height: 262px !important; flex: 0 0 auto !important; overflow: hidden; }}
[class*="st-key-card_ov_"] [data-testid="stElementToolbar"] {{ display: none; }}
[class*="st-key-card_ov_heat"], [class*="st-key-card_ov_list"] {{ height: 470px !important; }}
.ov-cap {{ font-size: 12px; color: var(--slate); margin: 0 2px 2px; line-height: 1.35; }}
.ov-foot {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px 16px;
  font-size: 11.5px; color: var(--slate); margin-top: 8px; }}
[class*="st-key-ov_rail"] label p {{ font-size: 12px; color: var(--slate); }}
@media (max-width: 720px) {{
  .ov-kpis .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }}
  [class*="st-key-card_ov_"] {{ height: auto !important; min-height: 260px; }}
}}
@media (max-width: 720px) {{ .nb-q {{ font-size: 28px; }} .nb-eyebrow {{ font-size: 10.5px; letter-spacing: .1em; }} }}
[data-testid="stMarkdownContainer"] ol, [data-testid="stMarkdownContainer"] ol li {{ font-size: 13.5px; line-height: 1.55; }}
</style>
"""


def inject_css() -> None:
    """Inject the stylesheet once per page run."""
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def info(text: str, align_right: bool = False) -> str:
    """Small circled i with a hover tooltip."""
    cls = "ci r" if align_right else "ci"
    return f'<span class="{cls}" data-def="{esc(text)}">i</span>'


def chip(text: str, tone: str = "neutral", tooltip: str | None = None) -> str:
    """Rounded mono chip; the tone picks the colors; an optional tooltip adds an i."""
    cls = normalize_tone(tone)
    tip = info(tooltip) if tooltip else ""
    return f'<span class="nb-chip {cls}">{esc(text)}{tip}</span>'


def header(
    eyebrow: str,
    question: str,
    summary: str,
    chips: Sequence[tuple[str, str | None]],
) -> str:
    """App header: eyebrow, question headline, one summary line and fact chips.

    Each chip is ``(text, tooltip_or_None)``.
    """
    chip_html = "".join(chip(t, "neutral", tip) for t, tip in chips)
    return (
        '<div class="nb-head">'
        f'<div class="nb-eyebrow">{esc(eyebrow)}</div>'
        f'<div class="nb-q">{esc(question)}</div>'
        f'<div class="nb-sum">{esc(summary)}</div>'
        f'<div class="nb-chips">{chip_html}</div>'
        "</div>"
    )


def bottom_line(
    question: str,
    answer: str,
    tone: str,
    sentences: Sequence[str],
    act: str | None = None,
    confidence: str | None = None,
    confidence_why: str | None = None,
    would_change: str | None = None,
) -> str:
    """The decision layer: question, colored answer chip, plain sentences, next step."""
    tone = normalize_tone(tone)
    color, tint = TONE_COLOR[tone], TONE_TINT[tone]
    body = "".join(f"<p>{esc(s)}</p>" for s in sentences)
    meta = ""
    if act:
        meta += (
            f'<div><div class="k">What to do</div><div class="v">{esc(act)}</div></div>'
        )
    if confidence:
        why = info(confidence_why) if confidence_why else ""
        meta += (
            '<div><div class="k">Confidence</div>'
            f'<div class="v">{esc(confidence)}{why}</div></div>'
        )
    if would_change:
        meta += (
            '<div><div class="k">What would change this</div>'
            f'<div class="v">{esc(would_change)}</div></div>'
        )
    meta_html = f'<div class="meta">{meta}</div>' if meta else ""
    return (
        '<div class="bl"><div class="lab">The bottom line</div>'
        f'<div class="q">{esc(question)}</div>'
        f'<div class="verd"><span class="ans" style="background:{tint};color:{color}">'
        f"{esc(answer)}</span></div>{body}{meta_html}</div>"
    )


def tile(
    value: str,
    label: str,
    tone: str = "neutral",
    tooltip: str | None = None,
    sub: str | None = None,
    bare: bool = False,
) -> str:
    """One big colored number with a label, optional tooltip and sub line."""
    tip = info(tooltip) if tooltip else ""
    sub_html = f'<div class="sub">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="tile{" bare" if bare else ""}">'
        f'<div class="num" style="color:{tone_color(tone)}">{esc(value)}</div>'
        f'<div class="lab">{esc(label)}{tip}</div>{sub_html}</div>'
    )


def tiles(items: Sequence[str], columns: int | None = None) -> str:
    """Lay out already-built tiles in a responsive grid (or a fixed column count)."""
    style = (
        f' style="grid-template-columns:repeat({int(columns)},minmax(0,1fr))"'
        if columns
        else ""
    )
    return f'<div class="tiles"{style}>{"".join(items)}</div>'


def takeaway(text: str) -> str:
    """One-sentence takeaway shown under (or above) a chart."""
    return f'<div class="takeaway">{esc(text)}</div>'


def card_title(text: str, tooltip: str | None = None) -> str:
    tip = info(tooltip) if tooltip else ""
    return f'<div class="cardtitle">{esc(text)}{tip}</div>'


def layer_label(text: str) -> str:
    return f'<div class="layer">{esc(text)}</div>'


def note(title: str, body: str, tone: str = "caution") -> str:
    """Callout box: amber for a known limitation, teal-tinted for a hint."""
    cls = "note accent" if tone == "accent" else "note"
    return f'<div class="{cls}"><b>{esc(title)}</b> {esc(body)}</div>'


@contextmanager
def card(name: str) -> Iterator[Any]:
    """Bordered card; ``name`` becomes a stable CSS key so the CSS can restyle it."""
    import streamlit as st

    with st.container(border=True, key=f"card_{name}") as c:
        yield c


# --------------------------------------------------------------------------- charts
def chart_config() -> dict[str, Any]:
    """Altair configuration: fonts, light gridlines, accent palette."""
    return {
        "background": PAPER,
        "view": {"stroke": None},
        "font": "Inter",
        "axis": {
            "labelFont": "Inter",
            "titleFont": "IBM Plex Mono",
            "labelColor": INK,
            "titleColor": SLATE,
            "labelFontSize": 12,
            "titleFontSize": 11,
            "titleFontWeight": "normal",
            "gridColor": "#EEF0F3",
            "domainColor": CARD_BORDER,
            "tickColor": CARD_BORDER,
        },
        "legend": {
            "labelFont": "Inter",
            "labelFontSize": 12,
            "labelColor": INK,
            "orient": "top",
            "symbolType": "square",
        },
        "range": {"category": [ACCENT, ACCENT_SOFT]},
    }


def style_chart(chart: Any, height: int | None = None) -> Any:
    """Apply the shared theme (and optional height) to an Altair chart."""
    if height is not None:
        chart = chart.properties(height=height)
    return chart.configure(**chart_config())
