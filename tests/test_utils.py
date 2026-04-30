"""Tests for fuzzyart.utils.math."""

import numpy as np
import pytest
from fuzzyart.utils.math import l1_norm, fuzzy_and, complement


class TestL1Norm:
    def test_positive(self):
        assert l1_norm(np.array([1.0, 2.0, 3.0])) == pytest.approx(6.0)

    def test_negative_values(self):
        assert l1_norm(np.array([-1.0, -2.0])) == pytest.approx(3.0)

    def test_zeros(self):
        assert l1_norm(np.zeros(5)) == pytest.approx(0.0)

    def test_returns_float(self):
        result = l1_norm(np.array([1.0, 2.0]))
        assert isinstance(result, float)


class TestFuzzyAnd:
    def test_element_wise_minimum(self):
        a = np.array([0.2, 0.8, 0.5])
        b = np.array([0.5, 0.3, 0.5])
        np.testing.assert_array_almost_equal(fuzzy_and(a, b), [0.2, 0.3, 0.5])

    def test_idempotent(self):
        a = np.array([0.3, 0.7])
        np.testing.assert_array_equal(fuzzy_and(a, a), a)

    def test_zero_dominates(self):
        a = np.array([0.9, 0.5])
        b = np.zeros(2)
        np.testing.assert_array_equal(fuzzy_and(a, b), np.zeros(2))


class TestComplement:
    def test_basic(self):
        v = np.array([0.2, 0.5, 1.0])
        np.testing.assert_array_almost_equal(complement(v), [0.8, 0.5, 0.0])

    def test_zero_maps_to_one(self):
        assert complement(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_double_complement_is_identity(self):
        v = np.array([0.1, 0.4, 0.9])
        np.testing.assert_array_almost_equal(complement(complement(v)), v)
