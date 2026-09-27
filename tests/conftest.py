"""Shared test data.

The tests download the Kaggle data with kagglehub, the same way the scripts
and the app do. It is a public dataset, so no Kaggle login is needed, and
kagglehub caches it after the first download.
"""

from pathlib import Path

import pytest

from data import load_big_leagues

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def big_leagues():
    return load_big_leagues()
