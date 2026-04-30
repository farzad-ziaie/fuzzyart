"""
Low-level fuzzy math primitives used throughout FuzzyARTMAP.

All functions operate on NumPy arrays and are kept stateless so they
can be used independently of any model class.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def l1_norm(vector: NDArray[np.float64]) -> float:
    """Compute the L1 (Manhattan) norm of a vector.

    Parameters
    ----------
    vector:
        1-D array of real values.

    Returns
    -------
    float
        Sum of absolute values: ``sum(|v_i|)``.

    Examples
    --------
    >>> l1_norm(np.array([1.0, -2.0, 3.0]))
    6.0
    """
    return float(np.sum(np.abs(vector)))


def fuzzy_and(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise fuzzy AND (minimum) of two vectors.

    In fuzzy set theory the AND operator is the element-wise minimum,
    equivalent to set intersection for fuzzy sets.

    Parameters
    ----------
    vec1, vec2:
        Arrays of the same shape with values in ``[0, 1]``.

    Returns
    -------
    NDArray
        ``min(vec1_i, vec2_i)`` for each position ``i``.

    Examples
    --------
    >>> fuzzy_and(np.array([0.2, 0.8]), np.array([0.5, 0.3]))
    array([0.2, 0.3])
    """
    return np.minimum(vec1, vec2)


def complement(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fuzzy complement (negation) of a vector.

    Parameters
    ----------
    vector:
        Array with values in ``[0, 1]``.

    Returns
    -------
    NDArray
        ``1 - vector``.

    Examples
    --------
    >>> complement(np.array([0.2, 0.5, 1.0]))
    array([0.8, 0.5, 0.0])
    """
    return 1.0 - vector
