"""
Fuzzy ARTMAP classifier.

Implementation of the Default ARTMAP algorithm described in:

    Carpenter, G.A. (2003). "Default ARTMAP."
    Proceedings of the International Joint Conference on Neural Networks.
    DOI: 10.1109/IJCNN.2003.1223900

Fuzzy ARTMAP is an incremental, online supervised learning algorithm.
It requires no pre-specification of the number of categories and can
learn new patterns without catastrophic forgetting of old ones.

Key properties
--------------
- **Online learning**: processes one sample at a time; can be called
  with ``partial_fit`` for true streaming use.
- **No forgetting**: committed nodes are never removed or overwritten
  in a destructive way.
- **Complement coding**: inputs are automatically encoded as
  ``[a, 1 - a]`` to preserve the L1 norm invariant.
- **Vigilance control**: the ``rho_baseline`` parameter controls the
  coarseness of the learned categories.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import trange

from fuzzyart.models.base import BaseART
from fuzzyart.preprocessing.transforms import complement_code, normalize
from fuzzyart.utils.math import fuzzy_and, l1_norm


class FuzzyARTMAP(BaseART):
    """Fuzzy ARTMAP supervised classifier.

    Parameters
    ----------
    alpha : float, default=0.01
        Signal rule parameter.  As ``alpha`` approaches zero the algorithm
        maximises code compression (fewer, more general nodes).
        Range: ``(0, ∞)``.
    beta : float, default=0.2
        Learning rate / learning fraction.  ``beta=1`` implements
        fast-commit / slow-recode learning.  Range: ``[0, 1]``.
    epsilon : float, default=-0.001
        Match-tracking parameter.  Negative values (MT-) allow coding
        of inconsistently labelled samples.  Range: ``(-1, 1)``.
    rho_baseline : float, default=0.0
        Baseline vigilance.  Higher values produce more specific (finer
        granularity) categories.  ``rho_baseline=0`` maximises
        compression.  Range: ``[0, 1]``.
    epochs : int, default=1
        Number of full passes over the training data.  ``epochs=1``
        simulates true online learning.
    verbose : bool, default=False
        If ``True``, show a progress bar during training.

    Attributes
    ----------
    n_committed_ : int
        Number of committed coding nodes (categories) after training.
    classes_ : NDArray
        Unique class labels encountered during training.
    W_ : list[NDArray]
        Weight vectors for each committed coding node.
        Shape of each element: ``(2 * n_features,)``.
    W_ab_ : list
        Class label assigned to each committed coding node.
    n_features_in_ : int
        Dimensionality of the **original** (pre-complement-coded) input.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from fuzzyart.models import FuzzyARTMAP
    >>> from fuzzyart.preprocessing import normalize
    >>> X, y = load_iris(return_X_y=True)
    >>> X = normalize(X)
    >>> clf = FuzzyARTMAP(alpha=0.01, beta=0.5, rho_baseline=0.0, epochs=5)
    >>> clf.fit(X, y)
    FuzzyARTMAP(alpha=0.01, beta=0.5, ...)
    >>> clf.predict(X[:5])
    array([0, 0, 0, 0, 0])
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.2,
        epsilon: float = -0.001,
        rho_baseline: float = 0.0,
        epochs: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.rho_baseline = rho_baseline
        self.epochs = epochs
        self.verbose = verbose

        # Runtime state — initialised in fit()
        self.W_: list[NDArray[np.float64]] = []
        self.W_ab_: list[Any] = []
        self.n_features_in_: int = 0
        self._M: int = 0  # 2 * n_features_in_ (complement-coded dim)
        self._rho: float = 0.0  # current vigilance (changes during search)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.float64], y: NDArray) -> "FuzzyARTMAP":
        """Train FuzzyARTMAP on labelled data.

        The input ``X`` should already be normalised to ``[0, 1]``.
        Complement coding is applied internally.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)`` with
            values in ``[0, 1]``.
        y:
            Class labels of shape ``(n_samples,)``.

        Returns
        -------
        self
        """
        X, y = self._validate_inputs(X, y)
        self._initialise(X, y)

        epoch_iter = trange(self.epochs, desc="Epochs", disable=not self.verbose)
        for epoch in epoch_iter:
            n_new = self._train_epoch(X, y, epoch)
            if self.verbose:
                epoch_iter.set_postfix(nodes=self.n_committed_, new=n_new)

        self.is_fitted_ = True
        return self

    def partial_fit(self, X: NDArray[np.float64], y: NDArray) -> "FuzzyARTMAP":
        """Incrementally train on a new batch — true online learning.

        Safe to call multiple times; existing nodes are never destroyed.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)`` with
            values in ``[0, 1]``.
        y:
            Class labels of shape ``(n_samples,)``.

        Returns
        -------
        self
        """
        X, y = self._validate_inputs(X, y)

        if not self.is_fitted_:
            self._initialise(X, y)
        else:
            # Merge any new classes seen in this batch
            self.classes_ = np.unique(np.concatenate([self.classes_, np.unique(y)]))

        self._train_epoch(X, y, epoch=0)
        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predict class labels for samples in ``X``.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)`` with
            values in ``[0, 1]``.

        Returns
        -------
        NDArray
            Predicted class labels of shape ``(n_samples,)``.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        A = complement_code(X)
        predictions = []
        for i in range(A.shape[0]):
            predictions.append(self._predict_one(A[i]))
        return np.array(predictions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, X: NDArray, y: NDArray
    ) -> tuple[NDArray[np.float64], NDArray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples; "
                f"got X: {X.shape[0]}, y: {y.shape[0]}."
            )
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError(
                "All values in X must be in [0, 1]. "
                "Use fuzzyart.preprocessing.normalize() first."
            )
        return X, y

    def _initialise(self, X: NDArray[np.float64], y: NDArray) -> None:
        """Set up weight matrices for first call to fit / partial_fit."""
        self.n_features_in_ = X.shape[1]
        self._M = 2 * self.n_features_in_
        # Collect all unique classes upfront so classes_ is complete after fit
        self.classes_ = np.unique(y)
        # Start with zero committed nodes; first sample always creates one
        self.W_ = []
        self.W_ab_ = []
        self.n_committed_ = 0

    def _train_epoch(
        self, X: NDArray[np.float64], y: NDArray, epoch: int
    ) -> int:
        """Run one epoch over the dataset. Returns number of new nodes."""
        A = complement_code(X)
        n_new = 0
        for i in range(A.shape[0]):
            added = self._train_one(A[i], y[i])
            if added:
                n_new += 1
        return n_new

    def _train_one(self, a: NDArray[np.float64], k: Any) -> bool:
        """Process a single complement-coded input ``a`` with label ``k``.

        Returns
        -------
        bool
            ``True`` if a new coding node was committed.
        """
        self._rho = self.rho_baseline

        # If no nodes yet, always commit
        if self.n_committed_ == 0:
            self._add_node(a, k)
            return True

        # Compute activation signals for all committed nodes
        T = self._compute_signals(a)

        # Sort nodes by activation (descending)
        order = np.flip(np.argsort(T))
        T_sorted = T[order]
        alpha_M = self.alpha * self._M
        candidates = order[T_sorted > alpha_M]

        if candidates.size == 0:
            self._add_node(a, k)
            return True

        # Search for a node that satisfies vigilance AND predicts k
        for j in candidates:
            match = l1_norm(fuzzy_and(a, self.W_[j])) / self._M
            if match >= self._rho:
                if self.W_ab_[j] == k or self.W_ab_[j] is None:
                    # Commit or update node
                    self.W_ab_[j] = k
                    self._update_weights(j, a)
                    return False
                else:
                    # Match-tracking: raise vigilance just above current match
                    self._rho = match + abs(self.epsilon)

        # No suitable node found — commit a new one
        self._add_node(a, k)
        return True

    def _predict_one(self, a: NDArray[np.float64]) -> Any:
        """Classify a single complement-coded input."""
        if self.n_committed_ == 0:
            return self.classes_[0] if len(self.classes_) > 0 else None

        T = self._compute_signals(a)
        order = np.flip(np.argsort(T))
        T_sorted = T[order]
        alpha_M = self.alpha * self._M
        candidates = order[T_sorted > alpha_M]

        if candidates.size == 0:
            # Fall back to highest activation node
            return self.W_ab_[int(np.argmax(T))]

        # Return the label of the highest-activation candidate
        return self.W_ab_[candidates[0]]

    def _compute_signals(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute activation signal T_j for all committed nodes.

        T_j = ||fuzzy_and(A, W_j)||_1 + (1 - alpha) * (M - ||W_j||_1)

        The second term acts as a tie-breaker: more committed (larger weight
        norm) nodes are preferred, which biases towards existing categories.
        """
        T = np.zeros(self.n_committed_)
        for j in range(self.n_committed_):
            numerator = l1_norm(fuzzy_and(a, self.W_[j]))
            tie_break = (1.0 - self.alpha) * (self._M - l1_norm(self.W_[j]))
            T[j] = numerator + tie_break
        return T

    def _update_weights(self, j: int, a: NDArray[np.float64]) -> None:
        """Slow-learning weight update for node ``j``.

        W_j ← β * fuzzy_and(A, W_j) + (1 − β) * W_j
        """
        self.W_[j] = (
            self.beta * fuzzy_and(a, self.W_[j])
            + (1.0 - self.beta) * self.W_[j]
        )

    def _add_node(self, a: NDArray[np.float64], k: Any) -> None:
        """Commit a new coding node with weight vector ``a`` and label ``k``."""
        self.W_.append(a.copy())
        self.W_ab_.append(k)
        self.n_committed_ += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_node_weights(self) -> NDArray[np.float64]:
        """Return a matrix of all committed node weight vectors.

        Returns
        -------
        NDArray
            Shape ``(n_committed_, 2 * n_features_in_)``.
        """
        self._check_is_fitted()
        return np.stack(self.W_, axis=0)

    def get_node_labels(self) -> list:
        """Return the class label assigned to each committed node.

        Returns
        -------
        list
            Length ``n_committed_``.
        """
        self._check_is_fitted()
        return list(self.W_ab_)

    def summary(self) -> dict:
        """Return a dictionary summarising the trained model.

        Returns
        -------
        dict
            Keys: ``n_committed``, ``n_features``, ``n_classes``,
            ``classes``, ``hyperparameters``.
        """
        self._check_is_fitted()
        return {
            "n_committed": self.n_committed_,
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "hyperparameters": self.get_params(),
        }
