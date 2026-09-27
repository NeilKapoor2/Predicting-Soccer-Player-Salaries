"""Scoring models and printing results in dollars.

Percent error is not used: it explodes for players with tiny salaries and
rewards a model that predicts near zero. Everything here is in dollars (MAE,
RMSE, MSE) or R².
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def report(y_true, y_pred):
    """The standard set of error measures for one model."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MSE": mse,
        "R2": r2_score(y_true, y_pred),
    }


def dollars(x):
    return f"${x:,.0f}"


def print_comparison(title, predictions, y_true):
    """One row per model: MAE, RMSE and R² on the same players.

    predictions is a dict of {model name: predicted salaries}.
    Returns the table as a DataFrame (MSE included) for later use.
    """
    table = pd.DataFrame({name: report(y_true, pred) for name, pred in predictions.items()}).T

    print(title)
    print("-" * 70)
    print(f"{'':30}{'MAE':>14}{'RMSE':>14}{'MSE':>14}{'R2':>8}")
    for name, row in table.iterrows():
        print(f"{name:30}{dollars(row['MAE']):>14}{dollars(row['RMSE']):>14}"
              f"{row['MSE']:>14.3e}{row['R2']:>8.2f}")
    print()
    return table


def print_biggest_misses(names, y_true, y_pred, n=10):
    """The players the model got most wrong, in both directions."""
    misses = pd.DataFrame({
        "Player": np.asarray(names),
        "Actual": np.asarray(y_true),
        "Predicted": np.asarray(y_pred),
    })
    misses["Miss"] = misses["Predicted"] - misses["Actual"]

    for label, rows in [("Paid far MORE than the model predicted", misses.nsmallest(n, "Miss")),
                        ("Paid far LESS than the model predicted", misses.nlargest(n, "Miss"))]:
        print(label)
        for _, row in rows.iterrows():
            print(f"  {row['Player']:28} actual {dollars(row['Actual']):>13}"
                  f"   predicted {dollars(row['Predicted']):>13}")
        print()
