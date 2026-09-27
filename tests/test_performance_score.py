import numpy as np
import pandas as pd
import pytest

from performance_score import MIN_90S, add_salary_gap, results_table, score_players, select_players
from conftest import REPO


@pytest.fixture(scope="module")
def players(big_leagues):
    return select_players(big_leagues)


def test_score_never_sees_salary(players):
    # The whole point of Project 3: remove salary entirely, or scramble it,
    # and every player's performance score must stay exactly the same.
    original = score_players(players)["performance_score"]

    without_salary = score_players(players.drop(columns=["Annual USD"]))["performance_score"]
    scrambled = players.copy()
    scrambled["Annual USD"] = np.random.default_rng(0).permutation(players["Annual USD"])

    pd.testing.assert_series_equal(original, without_salary)
    pd.testing.assert_series_equal(original, score_players(scrambled)["performance_score"])


def test_scores_are_between_0_and_100(players):
    scores = score_players(players)["performance_score"]
    assert scores.between(0, 100).all()


def test_only_eligible_players_are_scored(players):
    assert (players["Pos_x"] != "GK").all()
    assert (players["League"] != "Champions League").all()
    assert (players["90s"] >= MIN_90S).all()


def test_results_match_the_saved_csv(players):
    # If this fails, performance_score_results.csv (and the README) are out of date
    current = results_table(add_salary_gap(score_players(players))).reset_index(drop=True)
    saved = pd.read_csv(REPO / "performance_score_results.csv")
    pd.testing.assert_frame_equal(current, saved, check_dtype=False)
