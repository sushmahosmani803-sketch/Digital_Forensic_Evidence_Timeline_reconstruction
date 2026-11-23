"""Dashboard smoke tests using Streamlit's built-in AppTest runner."""

import py_compile

import pytest

from evidence.config import ROOT

APP = ROOT / "app.py"


def test_app_compiles():
    py_compile.compile(str(APP), doraise=True)


st = pytest.importorskip("streamlit")
pytest.importorskip("plotly")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def test_app_renders_without_errors(app):
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    text = " ".join(m.value for m in app.markdown)
    assert "Digital evidence timeline reconstruction" in text


def test_app_search_filters_events(app):
    app.text_input[0].input("wikileaks").run()
    assert not app.exception
    captions = " ".join(c.value for c in app.caption)
    assert "matching events" in captions


def test_app_switch_user_timeline(app):
    app.selectbox[0].select_index(3).run()
    assert not app.exception
