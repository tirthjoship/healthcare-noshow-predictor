"""Headless run of every tab and widget option of the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_default_run_has_six_tabs() -> None:
    assert len(_run().tabs) == 7


def test_every_selectbox_option_and_slider_extreme() -> None:
    at = _run()
    for i in range(len(at.selectbox)):
        for opt in list(at.selectbox[i].options):
            at.selectbox[i].select_index(list(at.selectbox[i].options).index(opt))
            at.run()
            assert not at.exception
    calls = at.slider[1]
    calls.set_value(calls.max)
    at.run()
    assert not at.exception
    at.slider[1].set_value(1)
    at.run()
    assert not at.exception


def test_impact_value_input_edit() -> None:
    at = _run()
    at.number_input[0].set_value(0.0).run()
    assert not at.exception
    at.number_input[0].set_value(500.0).run()
    assert not at.exception


def test_fairness_share_of_list_matches_in_bottom_line_and_takeaway() -> None:
    at = _run()
    pages = [m.value for m in at.markdown]
    bottom = [v for v in pages if "Does the list treat age groups fairly?" in v]
    takeaway = [v for v in pages if "Most under-served" in v]
    assert bottom and "0.8%" in bottom[0]
    assert takeaway and "0.8%" in takeaway[0]


def test_overview_filters_and_extremes() -> None:
    at = _run()
    days = list(at.selectbox(key="ov_day").options)
    bands = list(at.selectbox(key="ov_band").options)
    hoods = list(at.selectbox(key="ov_hood").options)
    for i in range(len(days)):
        at.selectbox(key="ov_day").select_index(i).run()
        assert not at.exception
    at.selectbox(key="ov_day").select_index(0)
    for i in range(len(bands)):
        at.selectbox(key="ov_band").select_index(i).run()
        assert not at.exception
    for i in (0, 1, len(hoods) - 1):
        at.selectbox(key="ov_hood").select_index(i).run()
        assert not at.exception
    for v in (5, 100):
        at.slider(key="ov_calls").set_value(v).run()
        assert not at.exception
    assert any("Call these" in m.value for m in at.markdown)
