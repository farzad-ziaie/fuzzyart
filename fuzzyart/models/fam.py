"""
Fuzzy ARTMAP classifier with optional input relevance weighting.

References
----------
Carpenter, G.A. (2003). "Default ARTMAP."
    IJCNN 2003. DOI: 10.1109/IJCNN.2003.1223900

Andonie, R. & Sasu, L. (2006). "Fuzzy ARTMAP with input relevances."
    IEEE Trans. Neural Networks, 17(4), 929-941.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import trange

from fuzzyart.models.base import BaseART
from fuzzyart.preprocessing.transforms import complement_code
from fuzzyart.utils.math import fuzzy_and, l1_norm


class FuzzyARTMAP(BaseART):
    """Fuzzy ARTMAP supervised classifier with optional input relevance.

    Parameters
    ----------
    alpha : float, default=0.01
        Signal rule parameter.  Range: ``(0, inf)``.
    beta : float, default=0.2
        Learning rate.  ``beta=1`` = fast learning.  Range: ``[0, 1]``.
    epsilon : float, default=-0.001
        Match-tracking parameter.  Range: ``(-1, 1)``.
    rho_baseline : float, default=0.0
        Baseline vigilance.  Higher = finer categories.  Range: ``[0, 1]``.
    relevance : NDArray or None, default=None
        Per-feature relevance weights ``r in [0,1]^(2M)`` applied to the
        complement-coded signals (Andonie & Sasu 2006).  ``None`` = uniform.
    epochs : int, default=1
    verbose : bool, default=False

    Attributes
    ----------
    n_committed_ : int
    classes_ : NDArray
    W_ : list[NDArray]  -- weight vectors shape (2*n_features,)
    W_ab_ : list        -- class label per node
    n_features_in_ : int
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.2,
        epsilon: float = -0.001,
        rho_baseline: float = 0.0,
        relevance: NDArray[np.float64] | None = None,
        epochs: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.rho_baseline = rho_baseline
        self.relevance = relevance
        self.epochs = epochs
        self.verbose = verbose

        self.W_: list[NDArray[np.float64]] = []
        self.W_ab_: list[Any] = []
        self.n_features_in_: int = 0
        self._M: int = 0
        self._r: NDArray | None = None
        self._rho: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, x: NDArray, y: NDArray) -> FuzzyARTMAP:
        """Train on labelled data.  X must be normalised to [0, 1]."""
        x, y = self._validate_inputs(x, y)
        self._initialise(x, y)
        epoch_iter = trange(self.epochs, desc="Epochs", disable=not self.verbose)
        for epoch in epoch_iter:
            n_new = self._train_epoch(x, y, epoch)
            if self.verbose:
                epoch_iter.set_postfix(nodes=self.n_committed_, new=n_new)
        self.is_fitted_ = True
        return self

    def partial_fit(self, x: NDArray, y: NDArray) -> FuzzyARTMAP:
        """Incremental / streaming fit.  Safe to call repeatedly."""
        x, y = self._validate_inputs(x, y)
        if not self.is_fitted_:
            self._initialise(x, y)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, np.unique(y)]))
        self._train_epoch(x, y, epoch=0)
        self.is_fitted_ = True
        return self

    def predict(self, x: NDArray) -> NDArray:
        """Return class labels for x.  x must be normalised to [0, 1]."""
        self._check_is_fitted()
        x = np.asarray(x, dtype=np.float64)
        a = complement_code(x)
        return np.array([self._predict_one(a[i]) for i in range(len(a))])

    def predict_proba(self, x: NDArray) -> NDArray:
        """Return class probability estimates for x.

        Probabilities are computed as a softmax over per-class activation
        sums: nodes are weighted by their activation signal, and each
        class receives the sum of activations for its nodes.

        Returns
        -------
        NDArray, shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        x = np.asarray(x, dtype=np.float64)
        a = complement_code(x)
        class_list = list(self.classes_)
        n_classes = len(class_list)
        proba = np.zeros((len(a), n_classes))
        for i in range(len(a)):
            t = self._compute_signals(a[i])
            # Accumulate activation per class
            class_act = np.zeros(n_classes)
            for j, label in enumerate(self.W_ab_):
                if label in class_list:
                    class_act[class_list.index(label)] += t[j]
            # Softmax
            class_act -= class_act.max()
            exp_act = np.exp(class_act)
            proba[i] = exp_act / exp_act.sum()
        return proba

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have the same number of samples; "
                f"got x: {x.shape[0]}, y: {y.shape[0]}."
            )
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError(
                "All values in x must be in [0, 1]. "
                "Use fuzzyart.preprocessing.normalize() first."
            )
        return x, y

    def _initialise(self, x: NDArray, y: NDArray) -> None:
        self.n_features_in_ = x.shape[1]
        self._M = 2 * self.n_features_in_
        self.classes_ = np.unique(y)
        self.W_ = []
        self.W_ab_ = []
        self.n_committed_ = 0
        # Resolve relevance vector
        if self.relevance is not None:
            r = np.asarray(self.relevance, dtype=np.float64)
            if r.shape != (self._M,):
                raise ValueError(
                    f"relevance must have shape ({self._M},) "
                    f"(= 2 * n_features); got {r.shape}."
                )
            self._r = r / (r.sum() + 1e-12) * self._M  # normalise so sum == M
        else:
            self._r = None

    def _train_epoch(self, x: NDArray, y: NDArray, epoch: int) -> int:
        a = complement_code(x)
        n_new = 0
        for i in range(len(a)):
            if self._train_one(a[i], y[i]):
                n_new += 1
        return n_new

    def _train_one(self, a: NDArray, k: Any) -> bool:
        """Returns True if a new node was committed."""
        self._rho = self.rho_baseline
        if self.n_committed_ == 0:
            self._add_node(a, k)
            return True

        t = self._compute_signals(a)
        order = np.flip(np.argsort(t))
        candidates = order[t[order] > self.alpha * self._M]

        if candidates.size == 0:
            self._add_node(a, k)
            return True

        for j in candidates:
            match = self._match_score(a, j)
            if match >= self._rho:
                if self.W_ab_[j] == k or self.W_ab_[j] is None:
                    self.W_ab_[j] = k
                    self._update_weights(j, a)
                    return False
                else:
                    self._rho = match + abs(self.epsilon)

        self._add_node(a, k)
        return True

    def _predict_one(self, a: NDArray) -> Any:
        if self.n_committed_ == 0:
            return self.classes_[0] if len(self.classes_) > 0 else None
        t = self._compute_signals(a)
        order = np.flip(np.argsort(t))
        candidates = order[t[order] > self.alpha * self._M]
        if candidates.size == 0:
            return self.W_ab_[int(np.argmax(t))]
        return self.W_ab_[candidates[0]]

    def _compute_signals(self, a: NDArray) -> NDArray:
        """T_j = ||r * fuzzy_and(A, W_j)||_1 + (1-alpha)*(||r||_1 - ||r*W_j||_1)

        When relevance r is uniform (None), this reduces to the standard
        Fuzzy ARTMAP signal rule.
        """
        w = np.stack(self.W_, axis=0)           # (C, M)
        fa = np.minimum(a[None, :], w)           # (C, M) vectorised fuzzy AND
        if self._r is not None:
            r = self._r
            numerator   = (r * fa).sum(axis=1)
            tie_break   = (1.0 - self.alpha) * (r.sum() - (r * w).sum(axis=1))
        else:
            numerator   = fa.sum(axis=1)
            tie_break   = (1.0 - self.alpha) * (self._M - w.sum(axis=1))
        return numerator + tie_break

    def _match_score(self, a: NDArray, j: int) -> float:
        fa = fuzzy_and(a, self.W_[j])
        if self._r is not None:
            return float((self._r * fa).sum() / self._r.sum())
        return float(l1_norm(fa) / self._M)

    def _update_weights(self, j: int, a: NDArray) -> None:
        self.W_[j] = self.beta * fuzzy_and(a, self.W_[j]) + (1.0 - self.beta) * self.W_[j]

    def _add_node(self, a: NDArray, k: Any) -> None:
        self.W_.append(a.copy())
        self.W_ab_.append(k)
        self.n_committed_ += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_node_weights(self) -> NDArray:
        self._check_is_fitted()
        return np.stack(self.W_, axis=0)

    def get_node_labels(self) -> list:
        self._check_is_fitted()
        return list(self.W_ab_)

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            "n_committed": self.n_committed_,
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "hyperparameters": self.get_params(),
        }
