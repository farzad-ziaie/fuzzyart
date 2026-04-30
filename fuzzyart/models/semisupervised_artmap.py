"""
Semi-supervised Bayesian ARTMAP.

Extends Bayesian ARTMAP to exploit *unlabelled* samples alongside labelled
ones.  Follows the framework of:

    Zhang, Y. et al. (2010). "Semi-supervised Bayesian ARTMAP."
    Neurocomputing, 73(16-18), 3001-3011.

Algorithm overview
------------------
1. **Supervised phase**: run Bayesian ARTMAP on labelled data to initialise
   category structure.
2. **EM phase** (repeated for ``em_iterations``):
   - **E-step**: for each unlabelled sample compute soft responsibilities
     P(j|x) across all categories.
   - **M-step**: update category means, variances, and class distributions
     using the weighted contributions from unlabelled data.
3. Class labels for unlabelled data are inferred from the posterior P(k|x).

Key benefits
------------
* Exploits cheap-to-obtain unlabelled medical data (e.g. unannotated scans)
* EM assignments are *soft*, not winner-take-all — reduces sensitivity to
  category initialisation
* Inherits Bayesian ARTMAP's calibrated uncertainty estimates
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import trange

from fuzzyart.models.bayesian_artmap import BayesianARTMAP


class SemiSupervisedARTMAP(BayesianARTMAP):
    """Semi-supervised Bayesian ARTMAP via EM.

    Parameters
    ----------
    rho_baseline : float, default=0.0
    epsilon : float, default=0.001
    initial_sigma : float, default=1.0
    max_sigma : float or None, default=None
    min_sigma : float, default=1e-6
    epochs : int, default=1
        Supervised training epochs on labelled data.
    em_iterations : int, default=10
        EM iterations applied to unlabelled data after supervised phase.
    unlabelled_weight : float, default=0.5
        Relative weight ``w in [0, 1]`` of unlabelled samples in the M-step
        update.  ``1.0`` treats unlabelled samples equally to labelled ones.
    verbose : bool, default=False

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from fuzzyart.models import SemiSupervisedARTMAP
    >>> from fuzzyart.preprocessing import normalize
    >>> X, y = load_iris(return_X_y=True)
    >>> X = normalize(X)
    >>> # Simulate partial labelling: only 30% labelled
    >>> rng = np.random.default_rng(0)
    >>> mask = rng.random(len(y)) < 0.3
    >>> clf = SemiSupervisedARTMAP(epochs=5, em_iterations=20)
    >>> clf.fit(X[mask], y[mask], X_unlabelled=X[~mask])
    SemiSupervisedARTMAP(...)
    """

    def __init__(
        self,
        rho_baseline: float = 0.0,
        epsilon: float = 0.001,
        initial_sigma: float = 1.0,
        max_sigma: float | None = None,
        min_sigma: float = 1e-6,
        epochs: int = 1,
        em_iterations: int = 10,
        unlabelled_weight: float = 0.5,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            rho_baseline=rho_baseline,
            epsilon=epsilon,
            initial_sigma=initial_sigma,
            max_sigma=max_sigma,
            min_sigma=min_sigma,
            epochs=epochs,
            verbose=verbose,
        )
        self.em_iterations = em_iterations
        self.unlabelled_weight = unlabelled_weight

    # ------------------------------------------------------------------
    # Public API — extends fit() to accept X_unlabelled
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        X_unlabelled: NDArray | None = None,
    ) -> "SemiSupervisedARTMAP":
        """Train using labelled data and (optionally) unlabelled data.

        Parameters
        ----------
        X : NDArray, shape (n_labelled, n_features)
            Labelled feature matrix in ``[0, 1]``.
        y : NDArray, shape (n_labelled,)
            Class labels.
        X_unlabelled : NDArray or None, shape (n_unlabelled, n_features)
            Unlabelled feature matrix in ``[0, 1]``.  If ``None``, falls
            back to standard Bayesian ARTMAP training.

        Returns
        -------
        self
        """
        # Phase 1: supervised training on labelled data
        super().fit(X, y)

        if X_unlabelled is None or len(X_unlabelled) == 0:
            return self

        X_u = np.asarray(X_unlabelled, dtype=np.float64)
        if np.any(X_u < 0) or np.any(X_u > 1):
            raise ValueError("X_unlabelled values must be in [0, 1].")

        # Phase 2: EM iterations
        em_iter = trange(
            self.em_iterations, desc="EM", disable=not self.verbose
        )
        for _ in em_iter:
            self._em_step(X_u)
            if self.verbose:
                em_iter.set_postfix(nodes=self.n_committed_)

        return self

    def predict_unlabelled(self, X_unlabelled: NDArray) -> NDArray:
        """Convenience: predict labels for unlabelled samples."""
        return self.predict(X_unlabelled)

    # ------------------------------------------------------------------
    # EM internals
    # ------------------------------------------------------------------

    def _em_step(self, X_u: NDArray) -> None:
        """One combined E+M step over all unlabelled samples."""
        class_list = list(self.classes_)
        total_n = sum(self.W_n_)

        # Accumulate soft statistics
        # mu_sum[j]  = sum_i gamma_ji * x_i
        # sq_sum[j]  = sum_i gamma_ji * x_i^2
        # n_sum[j]   = sum_i gamma_ji   (effective count)
        mu_sum   = [np.zeros(self._M) for _ in range(self.n_committed_)]
        sq_sum   = [np.zeros(self._M) for _ in range(self.n_committed_)]
        n_sum    = np.zeros(self.n_committed_)

        # E-step: compute responsibilities
        for x in X_u:
            log_joint = np.array([
                self._log_likelihood(x, j) + np.log(self.W_n_[j] / total_n)
                for j in range(self.n_committed_)
            ])
            log_joint -= log_joint.max()
            gamma = np.exp(log_joint)
            gamma /= gamma.sum() + 1e-300

            for j in range(self.n_committed_):
                w = float(gamma[j]) * self.unlabelled_weight
                mu_sum[j]  += w * x
                sq_sum[j]  += w * x ** 2
                n_sum[j]   += w

        # M-step: update category parameters
        for j in range(self.n_committed_):
            if n_sum[j] < 1e-10:
                continue
            n_total = self.W_n_[j] + n_sum[j]
            # Weighted mean: combine old labelled mean with soft unlabelled sum
            new_mu = (self.W_mu_[j] * self.W_n_[j] + mu_sum[j]) / n_total
            # Variance: E[X^2] - E[X]^2 from combined distribution
            old_ex2 = self.W_sigma_[j] ** 2 + self.W_mu_[j] ** 2
            new_ex2 = (old_ex2 * self.W_n_[j] + sq_sum[j]) / n_total
            new_var = np.maximum(new_ex2 - new_mu ** 2, self.min_sigma ** 2)
            self.W_mu_[j]    = new_mu
            self.W_sigma_[j] = np.sqrt(new_var)
            if self.max_sigma is not None:
                self.W_sigma_[j] = np.minimum(self.W_sigma_[j], self.max_sigma)
            # Keep W_n_ as the labelled count only (don't inflate with soft counts)
            # so that the supervised class distribution is not diluted
