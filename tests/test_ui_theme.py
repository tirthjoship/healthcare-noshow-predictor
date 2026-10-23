"""Unit tests for the pure markup helpers in ui_theme."""

from __future__ import annotations

import re

import ui_theme as ui


def test_esc_escapes_html_and_dollar() -> None:
    out = ui.esc('<script>alert("x")</script> $5 & $6')
    assert "<" not in out and ">" not in out and '"' not in out
    assert "$" not in out
    assert "&amp;" in out


def test_tone_mapping_known_and_unknown() -> None:
    assert ui.tone_color("good") == ui.GOOD
    assert ui.tone_color("caution") == ui.CAUTION
    assert ui.tone_color("problem") == ui.PROBLEM
    assert ui.tone_color("nonsense") == ui.INK
    assert ui.normalize_tone("nonsense") == "neutral"


def test_chip_tone_class_and_tooltip() -> None:
    assert 'class="nb-chip problem"' in ui.chip("x", "problem")
    assert "nb-chip neutral" in ui.chip("x", "bogus")
    assert 'data-def="a &amp; b"' in ui.chip("x", "good", "a & b")
    assert "data-def" not in ui.chip("x")


def test_header_escapes_and_contains_parts() -> None:
    out = ui.header("EYE", "Q <b>?", "sum", [("one", None), ("two", "tip")])
    assert "EYE" in out and "sum" in out
    assert "<b>" not in out
    assert out.count('class="nb-chip ') == 2
    assert 'data-def="tip"' in out


def test_bottom_line_structure_and_escaping() -> None:
    out = ui.bottom_line(
        "Q?",
        "<i>Yes</i>",
        "good",
        ["First.", "Second $5."],
        act="Do it",
        confidence="High",
        confidence_why="because",
        would_change="more data",
    )
    assert "<i>Yes</i>" not in out
    assert ui.TONE_COLOR["good"] in out
    assert out.count("<p>") == 2
    for label in ("What to do", "Confidence", "What would change this"):
        assert label in out
    assert 'data-def="because"' in out
    assert "$" not in out
    assert "\n" not in out


def test_bottom_line_optional_parts_omitted() -> None:
    out = ui.bottom_line("Q?", "A", "problem", ["S."])
    assert "meta" not in out
    assert ui.TONE_TINT["problem"] in out


def test_tile_color_and_sub() -> None:
    out = ui.tile("43%", "Hit <rate>", "good", "tip", sub="+25 pts")
    assert f"color:{ui.GOOD}" in out
    assert "&lt;rate&gt;" in out
    assert "+25 pts" in out
    grid = ui.tiles([out, out])
    assert grid.count('class="tile"') == 2


def test_no_newlines_so_markdown_keeps_html_block() -> None:
    for html_str in (
        ui.takeaway("t"),
        ui.card_title("t", "tip"),
        ui.layer_label("t"),
        ui.note("T", "body"),
        ui.header("e", "q", "s", []),
    ):
        assert "\n" not in html_str


def test_chart_config_uses_theme_fonts_and_palette() -> None:
    cfg = ui.chart_config()
    assert cfg["axis"]["labelFont"] == "Inter"
    assert cfg["range"]["category"][0] == ui.ACCENT
    assert re.fullmatch(r"#[0-9A-F]{6}", ui.ACCENT)


def test_tiles_columns_and_note_tone_and_bare_tile() -> None:
    assert "repeat(3," in ui.tiles(["a"], columns=3)
    assert "style=" not in ui.tiles(["a"])
    assert 'class="note accent"' in ui.note("T", "b", tone="accent")
    assert 'class="note"' in ui.note("T", "b")
    assert 'class="tile bare"' in ui.tile("1", "x", bare=True)


def test_author_footer_links_open_in_new_tab() -> None:
    out = ui.author_footer("A note.")
    assert "Built by Tirth Joshi" in out and out.endswith("A note.</div>")
    assert out.count('target="_blank" rel="noopener"') == 2
    assert "https://linkedin.com/in/tirthjoshi" in out
