"""Smoke tests: the Streamlit app runs and its filters work."""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import REPO
from performance_score import add_salary_gap, score_players, select_players
from salary_model import train_salary_model


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


# ----- Player view -----

@pytest.fixture(scope="module")
def scored(big_leagues):
    return add_salary_gap(score_players(select_players(big_leagues)))


def pick(row_id):
    app = run_app()
    app.selectbox[0].set_value(row_id).run()
    assert not app.exception
    return app


def metric(app, label):
    return next(m for m in app.metric if m.label == label)


def test_player_view_shows_the_same_numbers_as_the_table(scored):
    row_id = scored.index[scored["Player"] == "Jeremy Doku"][0]
    player = scored.loc[row_id]
    app = pick(row_id)
    assert metric(app, "Performance score").value == f"{player['performance_score']:.0f} / 100"
    assert metric(app, "Salary percentile").value == f"{player['salary_percentile']:.0f} / 100"
    assert metric(app, "Gap").value == f"{player['gap']:+.0f}"
    assert metric(app, "Actual salary").value == f"${player['Annual USD']:,.0f}"


def test_model_guess_matches_salary_model(big_leagues, scored):
    row_id = scored.index[scored["Player"] == "Jeremy Doku"][0]
    expected = train_salary_model(big_leagues).predict(scored.loc[row_id].to_dict())
    assert metric(pick(row_id), "Model's guess").value == f"${expected:,.0f}"


def test_repeated_names_can_be_told_apart(scored):
    app = run_app()
    labels = [app.selectbox[0].format_func(i) for i in scored.index[scored["Player"] == "Antony"]]
    assert len(set(labels)) == len(labels) == 2


def test_negative_guess_is_not_shown_as_a_salary(big_leagues, scored):
    model = train_salary_model(big_leagues)
    row_id = scored.apply(lambda row: model.predict(row.to_dict()), axis=1).idxmin()
    assert metric(pick(row_id), "Model's guess").value == "below $0"


def test_defenders_get_a_warning(scored):
    row_id = scored.index[scored["position_group"] == "DF"][0]
    assert any("can't see defending" in w.value for w in pick(row_id).warning)


def test_what_if_slider_changes_the_guess(scored):
    row_id = scored.index[scored["Player"] == "Jeremy Doku"][0]
    app = pick(row_id)
    before = metric(app, "Model's new guess").value
    app.slider(key=f"Gls-{row_id}").set_value(int(scored.loc[row_id, "Gls"]) + 10).run()
    assert not app.exception
    assert metric(app, "Model's new guess").value != before
