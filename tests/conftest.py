"""Shared pytest fixtures."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer


@pytest.fixture(scope="session")
def iris_normalized():
    """Iris dataset normalised to [0, 1]."""
    from fuzzyart.preprocessing import normalize
    X, y = load_iris(return_X_y=True)
    return normalize(X), y


@pytest.fixture(scope="session")
def binary_normalized():
    """Breast-cancer (binary) dataset normalised to [0, 1]."""
    from fuzzyart.preprocessing import normalize
    X, y = load_breast_cancer(return_X_y=True)
    return normalize(X), y


@pytest.fixture
def simple_X():
    """Tiny 3-sample, 2-feature dataset in [0, 1]."""
    return np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.5, 0.5],
    ])


@pytest.fixture
def simple_y():
    return np.array([0, 1, 0])
