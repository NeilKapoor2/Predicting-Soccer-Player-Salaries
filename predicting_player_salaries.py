"""Project 1: predicting MLS base salaries, and the data leakage test.

Five models predict base_salary from club, name, position and
guaranteed_compensation. Then the same models are trained again WITHOUT
guaranteed_compensation, on the same players, to measure how much they were
leaning on that one column (it is mostly base salary plus bonuses).

The step-by-step version, with my original exploration, is
Predicting_Player_Salaries.ipynb.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data import load_mls, mls_player_id, split_by_player
from evaluation import dollars, print_comparison, report

CATEGORICAL = ["club", "first_name", "last_name", "position"]
LEAKY_COLUMN = "guaranteed_compensation"

# name: (model, whether guaranteed_compensation should be scaled first).
# KNN and the neural network measure distances / use gradients, so they need
# scaled numbers. Trees and linear regression do not care.
MODELS = {
    "Linear Regression": (LinearRegression(), False),
    "Decision Tree": (DecisionTreeRegressor(max_depth=10, random_state=42), False),
    "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42), False),
    "KNN": (KNeighborsRegressor(n_neighbors=5), True),
    "Neural Network": (MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu",
                                    max_iter=500, random_state=42), True),
}


def make_pipeline(regressor, use_leaky_column, scale):
    transformers = [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    if use_leaky_column:
        transformers.append(("num", StandardScaler() if scale else "passthrough", [LEAKY_COLUMN]))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("regressor", clone(regressor)),
    ])


def train_all(X_train, y_train, use_leaky_column):
    """Fit every model in MODELS. Returns {name: fitted pipeline}."""
    fitted = {}
    for name, (regressor, scale) in MODELS.items():
        model = make_pipeline(regressor, use_leaky_column, scale)
        fitted[name] = model.fit(X_train, y_train)
    return fitted


def predict_all(models, X):
    """Predictions from every model, plus the ensemble (plain average of all five)."""
    predictions = {name: model.predict(X) for name, model in models.items()}
    predictions["Ensemble (average)"] = np.mean(list(predictions.values()), axis=0)
    return predictions


# ----- Data -----
df, rows_dropped = load_mls()
df = df[CATEGORICAL + [LEAKY_COLUMN, "base_salary", "year"]]
print(f"Player-seasons: {len(df)} ({rows_dropped} rows with missing values dropped)")

X = df[CATEGORICAL + [LEAKY_COLUMN]]
y = df["base_salary"]
player_id = mls_player_id(df)

# 80% train / 20% test, split by player (random_state=42)
train_idx, test_idx = split_by_player(player_id)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | no player in both\n")


# ----- Experiment 1: with guaranteed_compensation -----
models = train_all(X_train, y_train, use_leaky_column=True)
train_predictions = predict_all(models, X_train)
test_predictions = predict_all(models, X_test)

# Train vs test MSE: a big gap means the model memorised the training players
print(f"{'Mean squared error':30}{'Train':>14}{'Test':>14}")
for name in test_predictions:
    print(f"{name:30}{report(y_train, train_predictions[name])['MSE']:>14.3e}"
          f"{report(y_test, test_predictions[name])['MSE']:>14.3e}")
print()

with_gc = print_comparison("Test set, WITH guaranteed_compensation", test_predictions, y_test)


# ----- A player the models have never seen: Kevin De Bruyne -----
# He is not in MLS, so this shows what the models do outside their training data.
kevin = pd.DataFrame({
    "club": ["Manchester City"],
    "first_name": ["Kevin"],
    "last_name": ["De Bruyne"],
    "position": ["M"],
    LEAKY_COLUMN: [20800000],  # approximate annual compensation (USD)
})
print("Salary predictions for Kevin De Bruyne")
print("-" * 45)
for name, prediction in predict_all(models, kevin).items():
    print(f"{name:22}: ${prediction[0]:,.2f}")
print()


# ----- Experiment 2: without guaranteed_compensation -----
# Same train/test players as Experiment 1, so the only difference is the column.
X_train_no_gc = X_train.drop(columns=[LEAKY_COLUMN])
X_test_no_gc = X_test.drop(columns=[LEAKY_COLUMN])

models_no_gc = train_all(X_train_no_gc, y_train, use_leaky_column=False)
test_predictions_no_gc = predict_all(models_no_gc, X_test_no_gc)

without_gc = print_comparison("Test set, WITHOUT guaranteed_compensation",
                              test_predictions_no_gc, y_test)


# ----- The headline comparison -----
ensemble_with = with_gc.loc["Ensemble (average)"]
ensemble_without = without_gc.loc["Ensemble (average)"]

print("Ensemble error on the same held-out players")
print("-" * 62)
print(f"{'':34}{'MAE':>12}{'RMSE':>12}{'R2':>8}")
for label, row in [("With guaranteed compensation", ensemble_with),
                   ("Without guaranteed compensation", ensemble_without)]:
    print(f"{label:34}{dollars(row['MAE']):>12}{dollars(row['RMSE']):>12}{row['R2']:>8.2f}")
print("-" * 62)
print(f"MAE is {ensemble_without['MAE'] / ensemble_with['MAE']:.1f}x larger without guaranteed compensation")


# ----- Chart for the README -----
surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
labels = ["With guaranteed\ncompensation", "Without guaranteed\ncompensation"]
values = [ensemble_with["MAE"], ensemble_without["MAE"]]
ratio = values[1] / values[0]

fig, ax = plt.subplots(figsize=(6.5, 4.4), facecolor=surface)
ax.set_facecolor(surface)
bars = ax.bar(labels, values, width=0.45, color=["#2a78d6", "#eb6834"])

for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value, f"${value:,.0f}",
            ha="center", va="bottom", color=ink, fontsize=12, fontweight="bold")

direction = "larger" if ratio >= 1 else "smaller"
ax.set_title(f"Dropping one column made the error {max(ratio, 1 / ratio):.1f}x {direction}",
             loc="left", color=ink, fontsize=13, fontweight="bold", pad=22)
ax.text(0, 1.03, "Average salary error per player (MAE), same held-out players, ensemble of 5 models",
        transform=ax.transAxes, color=muted, fontsize=8.5)

ax.set_ylim(0, max(values) * 1.15)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.yaxis.grid(True, color=grid, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors=muted, length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(grid)

fig.savefig("leakage_comparison.png", dpi=200, bbox_inches="tight", facecolor=surface)
