import pandas as pd
import pytest

from flags import add_flags
from performance_score import add_salary_gap, score_players, select_players, stat_percentiles
from salary_model import train_salary_model
from scout import (FAIR, HIGH, LOW, MEDIUM, OVERPAID, UNCLEAR, UNDERPAID, percentile_among,
                   score_new_player, verdict)


@pytest.fixture(scope="module")
def scored(big_leagues):
    return add_salary_gap(score_players(select_players(big_leagues)))


def test_percentile_uses_the_same_rule_as_pandas():
    peers = [10, 20, 20, 30]
    everyone = pd.Series(peers + [20])
    assert percentile_among(peers, 20) == pytest.approx(everyone.rank(pct=True).iloc[-1] * 100)


@pytest.mark.parametrize("name", ["Jeremy Doku", "Zavier Gozo", "Romano Schmid"])
def test_unknown_player_scores_the_same_as_the_real_pipeline(scored, name):
    # Take a real player out of their peer group, score them as if they were
    # unknown, and compare with the performance score the pipeline gave them
    row_id = scored.index[scored["Player"] == name][0]
    player = scored.loc[row_id]
    group = scored[(scored["League"] == player["League"]) &
                   (scored["position_group"] == player["position_group"])]
    peers = group.drop(index=row_id)

    percentiles, score = score_new_player(player.to_dict(), peers)
    assert score == pytest.approx(player["performance_score"])
    for stat, value in stat_percentiles(scored).loc[row_id].items():
        assert percentiles[stat] == pytest.approx(value)
    assert percentile_among(peers["Annual USD"], player["Annual USD"]) == pytest.approx(player["salary_percentile"])


@pytest.mark.parametrize("gap, salary, guess, expected", [
    (50, 500_000, 7_000_000, (UNDERPAID, HIGH)),     # both strong
    (30, 2_000_000, 5_000_000, (UNDERPAID, MEDIUM)),  # both agree, not both strong
    (30, 4_000_000, 5_000_000, (UNDERPAID, LOW)),     # only the score sees it
    (0, 9_000_000, 5_000_000, (OVERPAID, LOW)),       # only the model sees it
    (-45, 12_000_000, 5_000_000, (OVERPAID, HIGH)),
    (30, 9_000_000, 5_000_000, (UNCLEAR, LOW)),       # they disagree
    (5, 5_500_000, 5_000_000, (FAIR, LOW)),           # nothing stands out
])
def test_verdict_and_confidence(gap, salary, guess, expected):
    assert verdict(gap, salary, guess, typical_miss=2_500_000) == expected


def test_high_confidence_is_rare_among_real_players(big_leagues, scored):
    model = train_salary_model(big_leagues)
    flagged = add_flags(scored, model)
    results = [verdict(g, s, p, model.test_mae)
               for g, s, p in zip(flagged["gap"], flagged["Annual USD"], flagged["model_guess"])]
    high = sum(confidence == HIGH for _, confidence in results)
    assert high == 38   # about 2.6% of 1,472, as the page says
