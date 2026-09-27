"""Loading the two Kaggle datasets and splitting them by player.

Every project in this repo uses these functions, so the data preparation and
the train/test split are written once and are identical everywhere.
"""

import os

import kagglehub
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MLS_DATASET = "crawford/us-major-league-soccer-salaries"
BIG_LEAGUES_DATASET = "armaanmartins21/undervalued-football-players"


def load_mls():
    """MLS salaries, one row per player-season, rows with missing values dropped.

    Used by Project 1 and split_comparison.py. Returns (df, rows_dropped).
    """
    path = kagglehub.dataset_download(MLS_DATASET)

    # One CSV per season. Store the season as a column.
    seasons = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            season = pd.read_csv(os.path.join(path, file))
            season["year"] = file.replace(".csv", "")
            seasons.append(season)

    df = pd.concat(seasons, ignore_index=True)
    rows_before = len(df)
    df = df.dropna()
    return df, rows_before - len(df)


def load_big_leagues():
    """Players from Europe's big leagues (plus MLS) with stats and salaries.

    Used by Projects 2 and 3. dfAll.csv in this repo is a reference copy.
    """
    path = kagglehub.dataset_download(BIG_LEAGUES_DATASET)
    csv_file = [file for file in os.listdir(path) if file.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))


def mls_player_id(df):
    """The MLS data has no player ID column, so use the full name."""
    return df["first_name"] + " " + df["last_name"]


def split_by_player(player_id, test_size=0.2, random_state=42):
    """Split row positions so every row of a player lands on the same side.

    Players appear in several seasons. A random split would put some of a
    player's seasons in training and others in testing, and the model could
    score well just by remembering names (see split_comparison.py).
    Returns (train_positions, test_positions).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(splitter.split(player_id, groups=player_id))

    # Sanity check: no player appears in both sets
    assert set(player_id.iloc[train]).isdisjoint(set(player_id.iloc[test]))
    return train, test
