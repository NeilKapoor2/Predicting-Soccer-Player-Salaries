"""Shared test data.

dfAll.csv is a byte-for-byte copy of the Kaggle file Projects 2 and 3 use, so
the tests run on real data without a Kaggle login (GitHub Actions has none).
"""

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def big_leagues():
    return pd.read_csv(REPO / "dfAll.csv")
