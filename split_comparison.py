"""Does the way I split train/test data change the MLS results?

Players show up in several seasons. A random split puts some of a player's
seasons in training and others in testing, so a model can score well just by
remembering names it has already seen. A player-based split (every season of a
player goes to the same side) removes that shortcut.

This script runs Linear Regression and Random Forest on both kinds of split,
with and without guaranteed_compensation, so the effect can be measured.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from data import load_mls, mls_player_id, split_by_player

# Same data as predicting_player_salaries.py
df, _ = load_mls()

X = df[["club", "first_name", "last_name", "position", "guaranteed_compensation"]]
y = df["base_salary"]
player_id = mls_player_id(df)

seasons_per_player = player_id.map(player_id.value_counts())
print(f"Rows: {len(df)} | unique players: {player_id.nunique()} | "
      f"rows belonging to players with more than one season: {int((seasons_per_player > 1).sum())}")

# Two ways to split the same rows, both with random_state=42
random_train, random_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
group_train, group_test = split_by_player(player_id)

for label, train, test in [("random split", random_train, random_test),
                           ("player split", group_train, group_test)]:
    seen = player_id.iloc[test].isin(set(player_id.iloc[train])).mean() * 100
    print(f"{label}: {seen:.1f}% of test rows are players who also appear in training")

categorical = ["club", "first_name", "last_name", "position"]


def evaluate(train, test, use_compensation):
    columns = categorical + (["guaranteed_compensation"] if use_compensation else [])
    transformers = [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]
    if use_compensation:
        transformers.append(("num", "passthrough", ["guaranteed_compensation"]))

    results = {}
    for name, regressor in [("Linear Regression", LinearRegression()),
                            ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]:
        model = Pipeline([("preprocessor", ColumnTransformer(transformers)), ("regressor", regressor)])
        model.fit(X.iloc[train][columns], y.iloc[train])
        predictions = model.predict(X.iloc[test][columns])
        results[name] = (r2_score(y.iloc[test], predictions), mean_absolute_error(y.iloc[test], predictions))
    return results


print()
print(f"{'':14}{'':16}{'Linear Regression':>26}{'Random Forest':>26}")
for label, train, test in [("random split", random_train, random_test),
                           ("player split", group_train, group_test)]:
    for use_compensation in (True, False):
        r = evaluate(train, test, use_compensation)
        tag = "with GC" if use_compensation else "without GC"
        lr, rf = r["Linear Regression"], r["Random Forest"]
        print(f"{label:14}{tag:16}"
              f"{f'R2 {lr[0]:.2f}, MAE ${lr[1]:,.0f}':>26}"
              f"{f'R2 {rf[0]:.2f}, MAE ${rf[1]:,.0f}':>26}")
