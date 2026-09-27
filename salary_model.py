"""Project 2's linear regression as plain numbers, for the Streamlit app.

A linear regression is just a starting value (the intercept) plus one weight
per stat, so a prediction is simple arithmetic:

    salary = intercept + weight_MP * MP + weight_Starts * Starts + ...

train_salary_model() trains it exactly like performance_based_salaries.py
(same rows, same split by player, same hyperparameter choice), so the app
shows the real Project 2 model, not a made-up formula. It also keeps the
model's test-set error, so the app can show how uncertain a prediction is.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import split_train_val_test

STATS = ["MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A",
         "xG", "xAG", "PrgC", "PrgP", "PrgR"]
TARGET = "Annual USD"


@dataclass
class SalaryModel:
    intercept: float
    weights: dict          # {stat: dollars per 1 unit of that stat}
    test_mae: float        # average miss on players the model never saw
    test_r2: float

    def predict(self, stats):
        """Predicted salary in dollars for one player.

        stats is a dict like {"MP": 30, "Min": 2400, "Gls": 10, ...}.
        G+A and 90s are always goals + assists and minutes / 90 in the data,
        so they are filled in here if missing. That keeps them consistent
        when a slider changes goals or minutes.
        """
        stats = dict(stats)
        stats.setdefault("G+A", stats["Gls"] + stats["Ast"])
        stats.setdefault("90s", round(stats["Min"] / 90, 1))
        return self.intercept + sum(self.weights[s] * stats[s] for s in STATS)


def fit_project2_linear_regression(df):
    """Train the linear regression exactly as Project 2 does.

    Returns (fitted sklearn model, X_test, y_test).
    """
    df = df.dropna(subset=STATS + [TARGET])
    X, y = df[STATS], df[TARGET]
    train, val, test = split_train_val_test(df["Player"])

    # Project 2 tries fit_intercept True and False and keeps the lower
    # validation MSE (on a tie the first one, True, wins)
    best_mse, best_model = float("inf"), None
    for fit_intercept in [True, False]:
        model = LinearRegression(fit_intercept=fit_intercept).fit(X.iloc[train], y.iloc[train])
        mse = mean_squared_error(y.iloc[val], model.predict(X.iloc[val]))
        if mse < best_mse:
            best_mse, best_model = mse, model
    return best_model, X.iloc[test], y.iloc[test]


def train_salary_model(df):
    """df is the big-leagues data, e.g. pd.read_csv("dfAll.csv")."""
    model, X_test, y_test = fit_project2_linear_regression(df)
    test_predictions = model.predict(X_test)
    return SalaryModel(
        intercept=float(model.intercept_),
        weights={stat: float(w) for stat, w in zip(STATS, model.coef_)},
        test_mae=mean_absolute_error(y_test, test_predictions),
        test_r2=r2_score(y_test, test_predictions),
    )


if __name__ == "__main__":
    # Print the model so you can see what each stat is "worth" to it
    salary_model = train_salary_model(pd.read_csv("dfAll.csv"))
    print(f"Starting value (intercept): ${salary_model.intercept:,.0f}")
    for stat, weight in salary_model.weights.items():
        print(f"  {stat:7} {weight:>+14,.0f} per 1")
    print(f"Typical miss on unseen players: ${salary_model.test_mae:,.0f} (R² {salary_model.test_r2:.2f})")
