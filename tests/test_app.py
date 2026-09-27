"""Smoke tests: the Streamlit app runs and its filters work."""

from streamlit.testing.v1 import AppTest

from conftest import REPO


def run_app():
    return AppTest.from_file(str(REPO / "app.py"), default_timeout=30).run()


def test_app_runs_without_errors():
    app = run_app()
    assert not app.exception
    assert len(app.dataframe) == 1


def test_league_filter_narrows_the_table():
    app = run_app()
    everyone = len(app.dataframe[0].value)

    league = app.multiselect[0].options[0]
    app.multiselect[0].select(league).run()
    filtered = app.dataframe[0].value

    assert not app.exception
    assert 0 < len(filtered) < everyone
    assert (filtered["League"] == league).all()
