"""
Ensemble wrappers for ART-family classifiers.

Two strategies are provided, both from the original Carpenter (1992) paper
and subsequent medical literature:

VotingARTMAP
    Trains ``n_voters`` classifiers on differently-shuffled orderings of
    the training data and combines predictions by majority vote (or
    probability averaging).  Reduces sensitivity to presentation order,
    which is the main source of variance in ARTMAP.

BaggingARTMAP
    Standard bootstrap-aggregating (Bagging) adapted for ARTMAP: each voter
    is trained on a bootstrap sample.  Reduces variance and gives a natural
    out-of-bag accuracy estimate.

Reference
---------
Carpenter, G.A. et al. (1992). "Fuzzy ARTMAP: A neural network architecture
    for incremental supervised learning of analog multidimensional maps."
    IEEE Trans. Neural Networks, 3(5), 698-713.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import trange

from fuzzyart.models.fam import FuzzyARTMAP


class VotingARTMAP(BaseEstimator, ClassifierMixin):
    """Ensemble of ARTMAP classifiers trained on shuffled orderings.

    Parameters
    ----------
    base_estimator : BaseART or None, default=None
        The base classifier to clone for each voter.  Defaults to
        ``FuzzyARTMAP()`` when ``None``.
    n_voters : int, default=5
        Number of independently trained classifiers.
    voting : {'hard', 'soft'}, default='hard'
        ``'hard'``: majority vote over predicted labels.
        ``'soft'``: average predicted probabilities (requires the base
        estimator to support ``predict_proba``).
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    voters_ : list
        Fitted voter instances.
    classes_ : NDArray
        Union of classes seen across all voters.

    Examples
    --------
    >>> from fuzzyart.models import VotingARTMAP, FuzzyARTMAP
    >>> from fuzzyart.preprocessing import normalize
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X = normalize(X)
    >>> ens = VotingARTMAP(base_estimator=FuzzyARTMAP(beta=0.5), n_voters=7)
    >>> ens.fit(X, y)
    VotingARTMAP(...)
    """

    def __init__(
        self,
        base_estimator=None,
        n_voters: int = 5,
        voting: str = "hard",
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_voters = n_voters
        self.voting = voting
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, x: NDArray, y: NDArray) -> VotingARTMAP:
        """Train ``n_voters`` classifiers on shuffled orderings of the data."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        base = FuzzyARTMAP() if self.base_estimator is None else self.base_estimator
        self.voters_: list = []
        self.classes_ = np.unique(y)

        voter_iter = trange(self.n_voters, desc="Voters", disable=not self.verbose)
        for _ in voter_iter:
            clf = clone(base)
            idx = rng.permutation(len(x))
            clf.fit(x[idx], y[idx])
            self.voters_.append(clf)
        return self

    def predict(self, x: NDArray) -> NDArray:
        """Return ensemble predictions."""
        check_is_fitted(self, "voters_")
        if self.voting == "soft":
            proba = self.predict_proba(x)
            idx = np.argmax(proba, axis=1)
            return np.array([self.classes_[i] for i in idx])
        # Hard voting: majority label per sample
        all_preds = np.stack([v.predict(x) for v in self.voters_], axis=0)  # (V, N)
        result = []
        for i in range(all_preds.shape[1]):
            labels, counts = np.unique(all_preds[:, i], return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.array(result)

    def predict_proba(self, x: NDArray) -> NDArray:
        """Return averaged class probabilities (requires soft-capable base)."""
        check_is_fitted(self, "voters_")
        class_list = list(self.classes_)
        n_classes = len(class_list)
        proba_sum = np.zeros((len(x), n_classes))
        for v in self.voters_:
            if hasattr(v, "predict_proba"):
                p = v.predict_proba(x)
                # Align columns in case a voter didn't see all classes
                voter_classes = list(v.classes_)
                for ki, klass in enumerate(class_list):
                    if klass in voter_classes:
                        proba_sum[:, ki] += p[:, voter_classes.index(klass)]
            else:
                # Fallback: one-hot encode hard predictions
                preds = v.predict(x)
                for ki, klass in enumerate(class_list):
                    proba_sum[:, ki] += (preds == klass).astype(float)
        return proba_sum / len(self.voters_)

    def get_voter_node_counts(self) -> list[int]:
        """Return the number of committed nodes in each voter."""
        check_is_fitted(self, "voters_")
        return [v.n_committed_ for v in self.voters_]


class BaggingARTMAP(BaseEstimator, ClassifierMixin):
    """Bootstrap-aggregated ARTMAP ensemble.

    Parameters
    ----------
    base_estimator : BaseART or None, default=None
        Base classifier.  Defaults to ``FuzzyARTMAP()``.
    n_estimators : int, default=10
        Number of bootstrap samples / classifiers.
    max_samples : float, default=1.0
        Fraction of training samples to draw (with replacement) for each
        estimator.  Range ``(0, 1]``.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    estimators_ : list
        Fitted classifiers.
    oob_score_ : float
        Out-of-bag accuracy (estimated without a separate test set).
    classes_ : NDArray
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, x: NDArray, y: NDArray) -> BaggingARTMAP:
        """Train on bootstrap samples and compute OOB score."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n = len(x)
        n_draw = max(1, int(n * self.max_samples))
        base = FuzzyARTMAP() if self.base_estimator is None else self.base_estimator
        self.classes_ = np.unique(y)
        self.estimators_: list = []
        self._oob_indices: list[NDArray] = []

        est_iter = trange(self.n_estimators, desc="Estimators", disable=not self.verbose)
        for _ in est_iter:
            idx = rng.integers(0, n, size=n_draw)
            oob = np.setdiff1d(np.arange(n), idx)
            clf = clone(base)
            clf.fit(x[idx], y[idx])
            self.estimators_.append(clf)
            self._oob_indices.append(oob)

        self.oob_score_ = self._compute_oob_score(x, y)
        return self

    def predict(self, x: NDArray) -> NDArray:
        check_is_fitted(self, "estimators_")
        all_preds = np.stack([e.predict(x) for e in self.estimators_], axis=0)
        result = []
        for i in range(all_preds.shape[1]):
            labels, counts = np.unique(all_preds[:, i], return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.array(result)

    def predict_proba(self, x: NDArray) -> NDArray:
        check_is_fitted(self, "estimators_")
        class_list = list(self.classes_)
        n_classes = len(class_list)
        proba_sum = np.zeros((len(x), n_classes))
        for e in self.estimators_:
            if hasattr(e, "predict_proba"):
                p = e.predict_proba(x)
                voter_classes = list(e.classes_)
                for ki, klass in enumerate(class_list):
                    if klass in voter_classes:
                        proba_sum[:, ki] += p[:, voter_classes.index(klass)]
            else:
                preds = e.predict(x)
                for ki, klass in enumerate(class_list):
                    proba_sum[:, ki] += (preds == klass).astype(float)
        return proba_sum / len(self.estimators_)

    def _compute_oob_score(self, x: NDArray, y: NDArray) -> float:
        """Estimate accuracy on OOB samples."""
        n = len(x)
        votes: dict[int, list] = {i: [] for i in range(n)}
        for clf, oob_idx in zip(self.estimators_, self._oob_indices, strict=True):
            if len(oob_idx) == 0:
                continue
            preds = clf.predict(x[oob_idx])
            for idx, pred in zip(oob_idx, preds, strict=True):
                votes[idx].append(pred)
        correct, total = 0, 0
        for i in range(n):
            if not votes[i]:
                continue
            labels, counts = np.unique(votes[i], return_counts=True)
            pred = labels[np.argmax(counts)]
            if pred == y[i]:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0
