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
    assert len(_run().tabs) == 6


def test_every_selectbox_option_and_slider_extreme() -> None:
    at = _run()
    for i in range(len(at.selectbox)):
        for opt in list(at.selectbox[i].options):
            at.selectbox[i].select_index(list(at.selectbox[i].options).index(opt))
            at.run()
            assert not at.exception
    at.slider[0].set_value(at.slider[0].max)
    at.run()
    assert not at.exception
    at.slider[0].set_value(1)
    at.run()
    assert not at.exception


def test_impact_value_input_edit() -> None:
    at = _run()
    at.number_input[0].set_value(0.0).run()
    assert not at.exception
    at.number_input[0].set_value(500.0).run()
    assert not at.exception
