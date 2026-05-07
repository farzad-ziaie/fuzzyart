"""
Abstract base class shared by all ART/ARTMAP variants.

Provides the sklearn-compatible interface (``fit`` / ``predict`` /
``get_params`` / ``set_params``) and the common state that every
ART-family model maintains.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseART(BaseEstimator, ClassifierMixin, ABC):
    """Sklearn-compatible abstract base for ART classifiers.

    All concrete ART models inherit from this class and must implement
    :meth:`fit` and :meth:`predict`.

    Attributes
    ----------
    n_committed_ : int
        Number of committed coding nodes after training.
    classes_ : NDArray
        Unique class labels seen during ``fit``.
    is_fitted_ : bool
        ``True`` after a successful call to ``fit``.
    """

    def __init__(self) -> None:
        self.n_committed_: int = 0
        self.classes_: NDArray | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, x: NDArray[np.float64], y: NDArray) -> BaseART:
        """Train the model on ``X`` with labels ``y``."""

    @abstractmethod
    def predict(self, x: NDArray[np.float64]) -> NDArray:
        """Return class predictions for ``X``."""

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the model to disk using pickle.

        Parameters
        ----------
        path:
            File path for the saved model (e.g. ``"model.pkl"``).
        """
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> BaseART:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path:
            Path to a ``.pkl`` file created by :meth:`save`.

        Returns
        -------
        BaseART
            The deserialised model instance.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # Sklearn compatibility helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({param_str})"
