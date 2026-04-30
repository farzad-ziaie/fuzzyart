"""
Input preprocessing transforms for FuzzyARTMAP.

FuzzyARTMAP requires all feature values to lie in ``[0, 1]``.
Complement coding doubles the feature space to preserve all
information after normalisation and guarantees constant L1 norm.

Reference
---------
Carpenter, G.A. et al. (1992) "Fuzzy ARTMAP: A neural network architecture
for incremental supervised learning of analog multidimensional maps."
IEEE Trans. Neural Networks, 3(5), 698-713.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD

from fuzzyart.utils.math import complement


def normalize(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale each feature column to ``[0, 1]`` by dividing by its maximum.

    Columns with a maximum of zero are left unchanged (division by one).

    Parameters
    ----------
    X:
        2-D array of shape ``(n_samples, n_features)``.

    Returns
    -------
    NDArray
        Normalised array of the same shape.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyart.preprocessing import normalize
    >>> normalize(np.array([[2.0, 4.0], [1.0, 8.0]]))
    array([[1. , 0.5],
           [0.5, 1. ]])
    """
    X = np.asarray(X, dtype=np.float64)
    maxes = np.abs(np.max(X, axis=0))
    maxes[maxes == 0] = 1.0
    return X / maxes


def complement_code(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply complement coding to produce a ``2M``-dimensional input.

    Concatenates each feature vector ``a`` with its complement ``1 - a``
    to produce ``A = [a, 1 - a]``.  This preserves the L1 norm across
    all inputs (``||A||_1 == M`` for normalised inputs).

    Parameters
    ----------
    X:
        2-D array of shape ``(n_samples, n_features)`` with values in
        ``[0, 1]``.

    Returns
    -------
    NDArray
        Array of shape ``(n_samples, 2 * n_features)``.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyart.preprocessing import complement_code
    >>> complement_code(np.array([[0.2, 0.8]]))
    array([[0.2, 0.8, 0.8, 0.2]])
    """
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X, complement(X)], axis=1)


def normalize_and_complement_code(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convenience: normalize then complement-code in one step.

    Parameters
    ----------
    X:
        2-D array of shape ``(n_samples, n_features)``.

    Returns
    -------
    NDArray
        Array of shape ``(n_samples, 2 * n_features)`` with values in
        ``[0, 1]``.
    """
    return complement_code(normalize(X))


def truncated_svd(X: NDArray[np.float64], n_components: int = 50) -> NDArray[np.float64]:
    """Reduce dimensionality via Truncated SVD (LSA).

    Useful for high-dimensional sparse inputs (e.g. text) before feeding
    into FuzzyARTMAP.

    Parameters
    ----------
    X:
        2-D array of shape ``(n_samples, n_features)``.
    n_components:
        Target number of dimensions.  Must be ``< min(n_samples, n_features)``.

    Returns
    -------
    NDArray
        Array of shape ``(n_samples, n_components)``.
    """
    svd = TruncatedSVD(n_components=n_components, n_iter=100)
    return svd.fit_transform(X)
