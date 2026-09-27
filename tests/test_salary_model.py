import numpy as np
import pytest

from salary_model import STATS, fit_project2_linear_regression, train_salary_model


@pytest.fixture(scope="module")
def salary_model(big_leagues):
    return train_salary_model(big_leagues)


def test_plain_numbers_match_scikit_learn_for_every_test_player(big_leagues, salary_model):
    # The app uses salary_model.predict (plain arithmetic). It has to give the
    # same answer as the trained scikit-learn model, to within a cent.
    sklearn_model, X_test, _ = fit_project2_linear_regression(big_leagues)
    expected = sklearn_model.predict(X_test)
    ours = [salary_model.predict(row) for row in X_test.to_dict("records")]
    np.testing.assert_allclose(ours, expected, rtol=0, atol=0.01)


def test_same_results_as_project_2(salary_model):
    # From performance_based_salaries.py: Linear Regression test MAE and R²
    assert salary_model.test_mae == pytest.approx(2_594_128.33, abs=1)
    assert salary_model.test_r2 == pytest.approx(0.1553, abs=1e-4)


def test_one_weight_per_stat(salary_model):
    assert list(salary_model.weights) == STATS


def test_goals_plus_assists_and_90s_are_filled_in(salary_model):
    player = {"MP": 30, "Starts": 28, "Min": 2430, "Gls": 12, "Ast": 7,
              "xG": 10.5, "xAG": 6.1, "PrgC": 60, "PrgP": 90, "PrgR": 180}
    complete = {**player, "G+A": 19, "90s": 27.0}
    assert salary_model.predict(player) == pytest.approx(salary_model.predict(complete))
