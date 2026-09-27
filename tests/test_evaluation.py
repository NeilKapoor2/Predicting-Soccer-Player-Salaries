import pytest

from evaluation import dollars, report


def test_perfect_predictions_have_zero_error():
    result = report([100_000, 250_000, 1_000_000], [100_000, 250_000, 1_000_000])
    assert result["MAE"] == 0
    assert result["RMSE"] == 0
    assert result["R2"] == 1


def test_errors_are_in_dollars():
    # Both predictions are off by $10,000
    result = report([100_000, 200_000], [110_000, 190_000])
    assert result["MAE"] == pytest.approx(10_000)
    assert result["RMSE"] == pytest.approx(10_000)
    assert result["MSE"] == pytest.approx(10_000 ** 2)
    assert result["R2"] == pytest.approx(0.96)


def test_predicting_the_average_gives_r2_of_zero():
    # This is why a negative R² means "worse than guessing the average"
    actual = [100_000, 200_000, 300_000]
    assert report(actual, [200_000] * 3)["R2"] == pytest.approx(0)


def test_dollars_format():
    assert dollars(1234567.89) == "$1,234,568"
