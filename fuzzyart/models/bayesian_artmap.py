"""
Bayesian ARTMAP classifier.

Replaces the hyperbox representation of Fuzzy ARTMAP with multivariate
Gaussian categories (diagonal covariance).  Key advantages over Fuzzy ARTMAP:

* **Calibrated probabilities** via ``predict_proba``
* **Reduced category proliferation** — Gaussian categories can grow/shrink
* **Uncertainty quantification** — entropy of the posterior distribution
* **Mahalanobis-based matching** — naturally handles different feature scales

Reference
---------
Vigdor, B. & Lerner, B. (2007). "The Bayesian ARTMAP."
    IEEE Trans. Neural Networks, 18(6), 1628-1644.
    DOI: 10.1109/TNN.2007.891195
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import trange

from fuzzyart.models.base import BaseART
from fuzzyart.preprocessing.transforms import normalize


class BayesianARTMAP(BaseART):
    """Bayesian ARTMAP with diagonal Gaussian categories.

    Each committed node ``j`` maintains a Gaussian distribution
    ``N(mu_j, diag(sigma_j^2))`` fitted online via Welford's algorithm.
    Category activation uses the log-posterior; vigilance is measured by
    the normalised Mahalanobis distance; predictions return calibrated
    class probabilities.

    Parameters
    ----------
    rho_baseline : float, default=0.0
        Baseline vigilance in ``[0, 1]``.  Corresponds to the minimum
        normalised match score ``exp(-0.5 * d_M^2 / M) >= rho``.
        Higher values enforce tighter category boundaries.
    epsilon : float, default=0.001
        Match-tracking increment (positive).  Vigilance is raised above
        ``current_match + epsilon`` when a mis-prediction occurs.
    initial_sigma : float, default=1.0
        Initial standard deviation for every dimension of a newly
        committed node.  Acts as a prior on category width.
    max_sigma : float or None, default=None
        Hard upper bound on per-dimension sigma (limits category size).
        ``None`` means unbounded.
    min_sigma : float, default=1e-6
        Minimum sigma — prevents degenerate zero-variance nodes.
    epochs : int, default=1
    verbose : bool, default=False

    Attributes
    ----------
    n_committed_ : int
    classes_ : NDArray
    W_mu_ : list[NDArray]      -- category means, shape (n_features,)
    W_sigma_ : list[NDArray]   -- category std devs (diagonal), shape (n_features,)
    W_n_ : list[int]           -- sample count per category
    W_class_ : list[dict]      -- class-label → count mapping per category
    n_features_in_ : int

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from fuzzyart import BayesianARTMAP
    >>> from fuzzyart.preprocessing import normalize
    >>> X, y = load_iris(return_X_y=True)
    >>> X = normalize(X)
    >>> clf = BayesianARTMAP(rho_baseline=0.1, epochs=5)
    >>> clf.fit(X, y)
    BayesianARTMAP(...)
    >>> clf.predict_proba(X[:2])
    array([[...]])
    """

    def __init__(
        self,
        rho_baseline: float = 0.0,
        epsilon: float = 0.001,
        initial_sigma: float = 1.0,
        max_sigma: float | None = None,
        min_sigma: float = 1e-6,
        epochs: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.rho_baseline = rho_baseline
        self.epsilon = epsilon
        self.initial_sigma = initial_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.epochs = epochs
        self.verbose = verbose

        self.W_mu_: list[NDArray[np.float64]] = []
        self.W_sigma_: list[NDArray[np.float64]] = []
        self.W_n_: list[int] = []
        self.W_class_: list[dict] = []
        self.n_features_in_: int = 0
        self._M: int = 0
        self._rho: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: NDArray, y: NDArray) -> "BayesianARTMAP":
        """Train on labelled data.  X must be normalised to [0, 1]."""
        X, y = self._validate_inputs(X, y)
        self._initialise(X, y)
        epoch_iter = trange(self.epochs, desc="Epochs", disable=not self.verbose)
        for epoch in epoch_iter:
            n_new = self._train_epoch(X, y)
            if self.verbose:
                epoch_iter.set_postfix(nodes=self.n_committed_, new=n_new)
        self.is_fitted_ = True
        return self

    def partial_fit(self, X: NDArray, y: NDArray) -> "BayesianARTMAP":
        """Incremental / streaming fit."""
        X, y = self._validate_inputs(X, y)
        if not self.is_fitted_:
            self._initialise(X, y)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, np.unique(y)]))
        self._train_epoch(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Return class labels for X."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.classes_[i] for i in idx])

    def predict_proba(self, X: NDArray) -> NDArray:
        """Return calibrated class probabilities P(k | x).

        Uses the full Bayesian posterior:
        ``P(k|x) = sum_j P(j|x) * P(k|j)``
        where ``P(j|x) ∝ p(x|j) * pi_j``.

        Returns
        -------
        NDArray, shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        n_classes = len(self.classes_)
        class_list = list(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, x in enumerate(X):
            proba[i] = self._posterior(x, class_list)
        return proba

    def predict_uncertainty(self, X: NDArray) -> NDArray:
        """Return predictive entropy H[P(k|x)] as an uncertainty measure.

        Higher entropy = less confident prediction.

        Returns
        -------
        NDArray, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        # Shannon entropy, safe log
        log_p = np.where(proba > 0, np.log(proba), 0.0)
        return -np.sum(proba * log_p, axis=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError(
                "Values must be in [0, 1]. Use fuzzyart.preprocessing.normalize()."
            )
        return X, y

    def _initialise(self, X: NDArray, y: NDArray) -> None:
        self.n_features_in_ = X.shape[1]
        self._M = X.shape[1]   # NOTE: no complement coding in Bayesian ARTMAP
        self.classes_ = np.unique(y)
        self.W_mu_ = []
        self.W_sigma_ = []
        self.W_n_ = []
        self.W_class_ = []
        self.n_committed_ = 0

    def _train_epoch(self, X: NDArray, y: NDArray) -> int:
        n_new = 0
        for i in range(len(X)):
            if self._train_one(X[i], y[i]):
                n_new += 1
        return n_new

    def _train_one(self, x: NDArray, k: Any) -> bool:
        """Returns True if a new node was committed."""
        self._rho = self.rho_baseline

        if self.n_committed_ == 0:
            self._add_node(x, k)
            return True

        # Activation: log p(x|j) + log pi_j  (log-posterior proportional)
        T = self._compute_log_activations(x)
        order = np.flip(np.argsort(T))

        for j in order:
            match = self._match_score(x, j)
            if match >= self._rho:
                if self._node_predicts(j, k):
                    self._update_node(j, x, k)
                    return False
                else:
                    self._rho = match + self.epsilon

        self._add_node(x, k)
        return True

    def _compute_log_activations(self, x: NDArray) -> NDArray:
        """log T_j = log p(x|j) + log pi_j"""
        total_n = sum(self.W_n_)
        T = np.zeros(self.n_committed_)
        for j in range(self.n_committed_):
            T[j] = self._log_likelihood(x, j) + np.log(self.W_n_[j] / total_n)
        return T

    def _log_likelihood(self, x: NDArray, j: int) -> float:
        """Diagonal Gaussian log-likelihood log N(x; mu_j, diag(sigma_j^2))."""
        diff = (x - self.W_mu_[j]) / self.W_sigma_[j]
        return -0.5 * (
            np.sum(diff ** 2) + np.sum(np.log(2 * np.pi * self.W_sigma_[j] ** 2))
        )

    def _match_score(self, x: NDArray, j: int) -> float:
        """Normalised Mahalanobis match: exp(-0.5 * d_M^2 / M) in [0, 1]."""
        diff = (x - self.W_mu_[j]) / self.W_sigma_[j]
        mahal_sq = float(np.sum(diff ** 2))
        return float(np.exp(-0.5 * mahal_sq / self._M))

    def _node_predicts(self, j: int, k: Any) -> bool:
        """Return True if node j's majority class == k, or node is new."""
        cd = self.W_class_[j]
        if not cd:
            return True
        majority = max(cd, key=cd.get)
        return majority == k

    def _update_node(self, j: int, x: NDArray, k: Any) -> None:
        """Welford's online update for mean and variance."""
        n_old = self.W_n_[j]
        self.W_n_[j] += 1
        n_new = self.W_n_[j]
        delta_old = x - self.W_mu_[j]
        self.W_mu_[j] += delta_old / n_new
        delta_new = x - self.W_mu_[j]
        if n_old >= 2:
            # Update running M2 approximation via variance update
            new_sigma_sq = (
                self.W_sigma_[j] ** 2 * n_old + delta_old * delta_new
            ) / n_new
            self.W_sigma_[j] = np.sqrt(np.maximum(new_sigma_sq, self.min_sigma ** 2))
        if self.max_sigma is not None:
            self.W_sigma_[j] = np.minimum(self.W_sigma_[j], self.max_sigma)
        # Update class distribution
        self.W_class_[j][k] = self.W_class_[j].get(k, 0) + 1

    def _add_node(self, x: NDArray, k: Any) -> None:
        self.W_mu_.append(x.copy())
        self.W_sigma_.append(np.full(self._M, float(self.initial_sigma)))
        self.W_n_.append(1)
        self.W_class_.append({k: 1})
        self.n_committed_ += 1

    def _posterior(self, x: NDArray, class_list: list) -> NDArray:
        """Compute P(k|x) over all classes."""
        n_classes = len(class_list)
        # Compute p(x|j) * pi_j for each node (in log space then exponentiate)
        total_n = sum(self.W_n_)
        log_joint = np.array([
            self._log_likelihood(x, j) + np.log(self.W_n_[j] / total_n)
            for j in range(self.n_committed_)
        ])
        # Stable softmax for node responsibilities P(j|x)
        log_joint -= log_joint.max()
        weights = np.exp(log_joint)
        weights /= weights.sum() + 1e-300

        # Marginalise over nodes: P(k|x) = sum_j P(j|x) * P(k|j)
        proba = np.zeros(n_classes)
        for j in range(self.n_committed_):
            cd = self.W_class_[j]
            n_j = self.W_n_[j]
            for ki, klass in enumerate(class_list):
                proba[ki] += weights[j] * cd.get(klass, 0) / n_j
        # Renormalise (should sum to ~1 already)
        s = proba.sum()
        return proba / (s + 1e-300)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_category_means(self) -> NDArray:
        """Return stacked category mean vectors, shape (n_committed_, n_features)."""
        self._check_is_fitted()
        return np.stack(self.W_mu_, axis=0)

    def get_category_sigmas(self) -> NDArray:
        """Return stacked diagonal std devs, shape (n_committed_, n_features)."""
        self._check_is_fitted()
        return np.stack(self.W_sigma_, axis=0)

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            "n_committed": self.n_committed_,
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "hyperparameters": self.get_params(),
        }
