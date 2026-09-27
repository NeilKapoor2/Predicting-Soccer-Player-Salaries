"""Project 2: predicting salary ONLY from on-field stats.

Players from Europe's big leagues (plus MLS). Five models are tuned on a
validation set, then combined two ways: a weighted average, and "stacking"
(a meta-model that learns how to combine the five predictions).

The step-by-step version, with my original exploration, is
Performance_Based_Salaries.ipynb.
"""

from itertools import product

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data import load_big_leagues, split_train_val_test
from evaluation import dollars, print_biggest_misses, print_comparison

STATS = ["MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A",
         "xG", "xAG", "PrgC", "PrgP", "PrgR"]
TARGET = "Annual USD"


def tune(name, make_model, grid, X_train, y_train, X_val, y_val, score=None):
    """Try every combination in grid and keep the one with the lowest validation MSE.

    grid is {parameter: [values to try]}. score(model) can replace the default
    "train on X_train, measure on X_val" (the meta network uses cross-validation).
    Returns (best parameters, best MSE).
    """
    best_mse, best_params = float("inf"), None
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = make_model(**params)
        if score is None:
            model.fit(X_train, y_train)
            mse = mean_squared_error(y_val, model.predict(X_val))
        else:
            mse = score(model)
        if mse < best_mse:
            best_mse, best_params = mse, params
    print(f"{name:20} best {best_params}  validation MSE {best_mse:.3e}")
    return best_params, best_mse


# ----- Data -----
df = load_big_leagues()
rows_before = len(df)
df = df.dropna(subset=STATS + [TARGET])
print(f"Rows: {len(df)} ({rows_before - len(df)} with missing stats or salary dropped)")

X = df[STATS]
y = df[TARGET]
player_id = df["Player"]

# Split by player: 64% train, 16% validation, 20% test (random_state=42)
train_idx, val_idx, test_idx = split_train_val_test(player_id)
X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

# No player appears in more than one set
assert set(player_id.loc[X_val.index]).isdisjoint(player_id.loc[X_test.index])
print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}\n")

# KNN and the neural network need scaled inputs. Fit the scaler on training
# data only, then apply the same scaling to validation and test.
scaler = StandardScaler().fit(X_train)
X_train_scaled, X_val_scaled, X_test_scaled = (scaler.transform(part) for part in (X_train, X_val, X_test))


# ----- Five base models, each tuned on the validation set -----
# name: (how to build it, hyperparameters to try, whether it needs scaled inputs)
MODELS = {
    "Linear Regression": (
        LinearRegression,
        {"fit_intercept": [True, False]},
        False,
    ),
    "Decision Tree": (
        lambda **p: DecisionTreeRegressor(random_state=42, **p),
        {"max_depth": [3, 5, 10, 15, 20, None],
         "min_samples_split": [2, 5, 10],
         "min_samples_leaf": [1, 2, 5]},
        False,
    ),
    "Random Forest": (
        lambda **p: RandomForestRegressor(random_state=42, **p),
        {"n_estimators": [50, 100, 200],
         "max_depth": [5, 10, 15, None],
         "min_samples_split": [2, 5, 10],
         "min_samples_leaf": [1, 2, 5]},
        False,
    ),
    "KNN": (
        KNeighborsRegressor,
        {"n_neighbors": [3, 5, 10, 15, 20],
         "weights": ["uniform", "distance"],
         "p": [1, 2]},
        True,
    ),
    "Neural Network": (
        lambda **p: MLPRegressor(max_iter=1000, random_state=42, **p),
        {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
         "learning_rate_init": [0.001, 0.01],
         "alpha": [0.0001, 0.001, 0.01]},
        True,
    ),
}

val_predictions, test_predictions, val_mse = {}, {}, {}
print("Hyperparameter search")
for name, (make_model, grid, scaled) in MODELS.items():
    X_tr, X_va, X_te = (X_train_scaled, X_val_scaled, X_test_scaled) if scaled else (X_train, X_val, X_test)
    best_params, val_mse[name] = tune(name, make_model, grid, X_tr, y_train, X_va, y_val)

    model = make_model(**best_params).fit(X_tr, y_train)
    val_predictions[name] = model.predict(X_va)
    test_predictions[name] = model.predict(X_te)
print()


# ----- Combination 1: weighted average -----
# Each model's weight is proportional to 1 / validation MSE, so better models count more.
inverse_mse = np.array([1 / val_mse[name] for name in MODELS])
weights = inverse_mse / inverse_mse.sum()
print("Weighted ensemble weights:",
      ", ".join(f"{name} {w:.3f}" for name, w in zip(MODELS, weights)))

all_predictions = dict(test_predictions)
all_predictions["Weighted Ensemble"] = sum(w * test_predictions[name] for name, w in zip(MODELS, weights))


# ----- Combination 2: stacking -----
# The meta-model's inputs are the five models' predictions, and it learns the
# best way to combine them. It is trained on the validation set, because the
# base models never saw those players.
meta_X_train = np.column_stack([val_predictions[name] for name in MODELS])
meta_X_test = np.column_stack([test_predictions[name] for name in MODELS])

# Linear meta-model
meta_linear = LinearRegression().fit(meta_X_train, y_val)
all_predictions["Stacked (linear)"] = meta_linear.predict(meta_X_test)
print("Linear meta-model coefficients:",
      ", ".join(f"{name} {c:.4f}" for name, c in zip(MODELS, meta_linear.coef_)),
      f"| intercept {meta_linear.intercept_:,.0f}")

# Neural-network meta-model
meta_scaler = StandardScaler().fit(meta_X_train)
meta_X_train_scaled = meta_scaler.transform(meta_X_train)
meta_X_test_scaled = meta_scaler.transform(meta_X_test)


def make_meta_nn(**mlp_params):
    # Salaries are in the millions, so the target has to be scaled as well as
    # the inputs. Without this the network stays stuck near zero
    # (this was the bug that made the old meta network predict about $100).
    return TransformedTargetRegressor(
        regressor=MLPRegressor(
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            **mlp_params
        ),
        transformer=StandardScaler()
    )


def cross_validated_mse(model):
    # Only about 450 validation rows, so score with 5-fold cross-validation
    # instead of on the rows the network was trained on
    return -cross_val_score(model, meta_X_train_scaled, y_val,
                            scoring="neg_mean_squared_error", cv=5).mean()


best_params, _ = tune(
    "Meta Neural Network", make_meta_nn,
    {"hidden_layer_sizes": [(10,), (20,), (10, 10), (20, 10)],
     "learning_rate_init": [0.001, 0.01],
     "alpha": [0.0001, 0.001, 0.01]},
    None, None, None, None, score=cross_validated_mse,
)
meta_nn = make_meta_nn(**best_params).fit(meta_X_train_scaled, y_val)
all_predictions["Meta Neural Network"] = meta_nn.predict(meta_X_test_scaled)
print()


# ----- Results -----
table = print_comparison("Test set (players no model has seen)", all_predictions, y_test)
best = table["MAE"].idxmin()
print(f"Best model by MAE: {best} ({dollars(table.loc[best, 'MAE'])})\n")

print("Random Forest, biggest misses on the test set")
print_biggest_misses(df.loc[X_test.index, "Player"], y_test, test_predictions["Random Forest"])
