import pytest

from flags import MIXED, NO_FLAG, OVERPAID, UNDERPAID, add_flags, combine, model_signal, score_signal
from performance_score import add_salary_gap, score_players, select_players
from salary_model import train_salary_model


def test_score_signal_uses_the_threshold():
    assert score_signal(25) == 1
    assert score_signal(24.9) == 0
    assert score_signal(-25) == -1
    assert score_signal(0) == 0


def test_model_signal_is_about_the_error_range():
    # Model guesses $5M and is typically off by $2M: the range is $3M to $7M
    assert model_signal(2_000_000, 5_000_000, 2_000_000) == 1    # paid below the range
    assert model_signal(8_000_000, 5_000_000, 2_000_000) == -1   # paid above the range
    assert model_signal(4_000_000, 5_000_000, 2_000_000) == 0    # inside the range


@pytest.mark.parametrize("score, model, flag", [
    (1, 1, UNDERPAID), (-1, -1, OVERPAID),
    (1, -1, MIXED), (-1, 1, MIXED),
    (1, 0, NO_FLAG), (0, -1, NO_FLAG), (0, 0, NO_FLAG),
])
def test_a_flag_needs_both_methods_to_agree(score, model, flag):
    assert combine(score, model) == flag


def test_flag_counts(big_leagues):
    scores = add_salary_gap(score_players(select_players(big_leagues)))
    flagged = add_flags(scores, train_salary_model(big_leagues))
    assert flagged["flag"].value_counts().to_dict() == {
        NO_FLAG: 1234, UNDERPAID: 105, OVERPAID: 105, MIXED: 28,
    }
