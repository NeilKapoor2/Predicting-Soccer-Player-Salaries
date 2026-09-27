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


def test_all_players_tab_shows_everyone_even_when_filtered():
    app = run_app()
    everyone = len(app.dataframe[0].value)
    app.multiselect[0].select(app.multiselect[0].options[0]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == everyone == 1472


# ----- Player view -----

@pytest.fixture(scope="module")
def scored(big_leagues):
    return add_salary_gap(score_players(select_players(big_leagues)))


def pick(row_id):
    app = run_app()
    app.selectbox(key="player-picker").set_value(row_id).run()
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
    labels = [app.selectbox(key="player-picker").format_func(i) for i in scored.index[scored["Player"] == "Antony"]]
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


def test_picker_and_filters_are_in_the_lookup_tab_not_the_left_pane():
    app = run_app()
    assert len(app.sidebar.selectbox) == 0 and len(app.sidebar.multiselect) == 0
    assert app.selectbox(key="player-picker") is not None


def test_league_filter_narrows_the_player_list(scored):
    app = run_app()
    league = app.multiselect[0].options[0]
    app.multiselect[0].select(league).run()
    offered = app.selectbox(key="player-picker").options
    assert len(offered) == (scored["League"] == league).sum()


def test_player_view_shows_the_flag(scored):
    row_id = scored.index[scored["Player"] == "Zavier Gozo"][0]
    app = pick(row_id)
    assert any("possibly underpaid" in block.value.lower() for block in app.markdown)


def test_flag_filter_narrows_the_list():
    app = run_app()
    flag_filter = next(box for box in app.multiselect if box.label == "Scouting flag")
    flag_filter.select("Possibly underpaid").run()
    assert not app.exception
    assert len(app.selectbox(key="player-picker").options) == 105


# ----- Scout a new player -----

def test_scout_tab_gives_a_verdict():
    app = run_app()
    assert not app.exception
    assert any("confidence" in block.value for block in app.markdown)


def test_scout_low_salary_star_forward_is_underpaid():
    app = run_app()
    app.selectbox(key="scout-position").set_value("FW").run()
    k = "Premier League-FW"
    for stat, value in [("Gls", 25), ("Ast", 12), ("xG", 22.0), ("xAG", 10.0), ("Min", 3000), ("MP", 36), ("Starts", 34)]:
        app.slider(key=f"scout-{stat}-{k}").set_value(value)
    app.number_input(key=f"scout-salary-{k}").set_value(300_000).run()
    assert not app.exception
    stamp = next(block.value for block in app.markdown if "confidence</span>" in block.value)
    assert ">Underpaid<" in stamp


def test_all_players_tab_has_its_own_filters():
    app = run_app()
    app.multiselect(key="all-flag").select("Possibly underpaid").run()
    assert not app.exception
    table = app.dataframe[0].value
    assert len(table) == 105 and (table["flag"] == "Possibly underpaid").all()
    # The lookup tab's player list is unaffected
    assert len(app.selectbox(key="player-picker").options) == 1472


def test_ordinal_suffixes():
    source = (REPO / "app.py").read_text()
    namespace = {}
    exec(source[source.index("def ordinal"):source.index("def millions")], namespace)
    ordinal = namespace["ordinal"]
    assert [ordinal(n) for n in [1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 70, 100, 3.4]] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "70th", "100th", "3rd"]
