"""Tests for fuzzyart.preprocessing.transforms."""

import numpy as np
import pytest

from fuzzyart.preprocessing import (
    complement_code,
    normalize,
    normalize_and_complement_code,
)


class TestNormalize:
    def test_max_becomes_one(self):
        x = np.array([[2.0, 4.0], [1.0, 8.0]])
        out = normalize(x)
        assert out.max() == pytest.approx(1.0)

    def test_shape_preserved(self):
        x = np.random.rand(10, 5) * 10
        assert normalize(x).shape == x.shape

    def test_zero_column_unchanged(self):
        x = np.array([[0.0, 1.0], [0.0, 2.0]])
        out = normalize(x)
        np.testing.assert_array_equal(out[:, 0], [0.0, 0.0])

    def test_values_in_unit_interval(self):
        x = np.random.rand(20, 8) * 100
        out = normalize(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-10


class TestComplementCode:
    def test_doubles_features(self):
        x = np.array([[0.2, 0.8]])
        assert complement_code(x).shape == (1, 4)

    def test_complement_correct(self):
        x = np.array([[0.2, 0.8]])
        out = complement_code(x)
        np.testing.assert_array_almost_equal(out, [[0.2, 0.8, 0.8, 0.2]])

    def test_l1_norm_constant(self):
        """||[a, 1-a]||_1 == n_features for any normalised input."""
        np.random.seed(42)
        x = np.random.rand(50, 6)
        a = complement_code(x)
        norms = np.sum(a, axis=1)
        np.testing.assert_array_almost_equal(norms, np.full(50, 6.0))

    def test_values_in_unit_interval(self):
        x = np.random.rand(10, 4)
        out = complement_code(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestNormalizeAndComplementCode:
    def test_pipeline_equivalent(self):
        x = np.random.rand(15, 5) * 10
        expected = complement_code(normalize(x))
        np.testing.assert_array_almost_equal(
            normalize_and_complement_code(x), expected
        )
