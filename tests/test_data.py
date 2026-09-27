import pandas as pd

from data import mls_player_id, split_by_player, split_train_val_test


def test_no_player_is_in_both_train_and_test(big_leagues):
    player_id = big_leagues["Player"]
    train, test = split_by_player(player_id)
    assert set(player_id.iloc[train]).isdisjoint(player_id.iloc[test])


def test_every_row_is_used_exactly_once(big_leagues):
    train, test = split_by_player(big_leagues["Player"])
    assert sorted(list(train) + list(test)) == list(range(len(big_leagues)))


def test_about_20_percent_of_players_are_held_out(big_leagues):
    player_id = big_leagues["Player"]
    _, test = split_by_player(player_id)
    share = player_id.iloc[test].nunique() / player_id.nunique()
    assert 0.19 < share < 0.21


def test_split_is_the_same_every_run(big_leagues):
    first = split_by_player(big_leagues["Player"])
    second = split_by_player(big_leagues["Player"])
    assert (first[0] == second[0]).all() and (first[1] == second[1]).all()


def test_project_2_three_way_split_has_no_shared_players(big_leagues):
    player_id = big_leagues["Player"]
    train, val, test = split_train_val_test(player_id)

    train_players = set(player_id.iloc[train])
    val_players = set(player_id.iloc[val])
    test_players = set(player_id.iloc[test])
    assert train_players.isdisjoint(val_players)
    assert train_players.isdisjoint(test_players)
    assert val_players.isdisjoint(test_players)
    assert len(train) + len(val) + len(test) == len(big_leagues)


def test_mls_player_id_is_the_full_name():
    df = pd.DataFrame({"first_name": ["David", "Landon"], "last_name": ["Beckham", "Donovan"]})
    assert list(mls_player_id(df)) == ["David Beckham", "Landon Donovan"]
