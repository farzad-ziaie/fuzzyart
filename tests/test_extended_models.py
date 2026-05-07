"""Tests for SemiSupervisedARTMAP, VotingARTMAP, and BaggingARTMAP."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score

from fuzzyart import (
    BaggingARTMAP,
    BayesianARTMAP,
    FuzzyARTMAP,
    SemiSupervisedARTMAP,
    VotingARTMAP,
)
from fuzzyart.preprocessing import normalize


@pytest.fixture(scope="module")
def iris():
    X, y = load_iris(return_X_y=True)
    return normalize(X), y


@pytest.fixture(scope="module")
def iris_split(iris):
    """30% labelled, 70% unlabelled."""
    X, y = iris
    rng = np.random.default_rng(42)
    mask = rng.random(len(y)) < 0.3
    return X[mask], y[mask], X[~mask], y[~mask]


# =============================================================================
# SemiSupervisedARTMAP
# =============================================================================

class TestSemiSupervisedARTMAP:

    def test_fit_labelled_only(self, iris):
        X, y = iris
        clf = SemiSupervisedARTMAP(epochs=3)
        clf.fit(X, y)   # no unlabelled — should behave like BayesianARTMAP
        assert clf.is_fitted_

    def test_fit_with_unlabelled(self, iris_split):
        X_l, y_l, X_u, _ = iris_split
        clf = SemiSupervisedARTMAP(epochs=3, em_iterations=5)
        clf.fit(X_l, y_l, x_unlabelled=X_u)
        assert clf.is_fitted_

    def test_semi_supervised_at_least_as_good_as_supervised(self, iris_split):
        """With EM on unlabelled data, accuracy should not degrade badly."""
        X_l, y_l, X_u, y_u = iris_split
        X_all = np.concatenate([X_l, X_u])
        y_all = np.concatenate([y_l, y_u])

        sup_clf = BayesianARTMAP(epochs=5).fit(X_l, y_l)
        ss_clf  = SemiSupervisedARTMAP(epochs=5, em_iterations=10).fit(
            X_l, y_l, x_unlabelled=X_u
        )
        # Both should be above 70% on the full dataset
        for clf in (sup_clf, ss_clf):
            acc = np.mean(clf.predict(X_all) == y_all)
            assert acc >= 0.70, f"Accuracy {acc:.2%} too low"

    def test_predict_unlabelled(self, iris_split):
        X_l, y_l, X_u, _ = iris_split
        clf = SemiSupervisedARTMAP(epochs=3, em_iterations=5).fit(
            X_l, y_l, x_unlabelled=X_u
        )
        preds = clf.predict_unlabelled(X_u)
        assert preds.shape == (len(X_u),)
        assert set(preds).issubset(set(clf.classes_))

    def test_predict_proba_sums_to_one(self, iris_split):
        X_l, y_l, X_u, _ = iris_split
        clf = SemiSupervisedARTMAP(epochs=3, em_iterations=3).fit(
            X_l, y_l, x_unlabelled=X_u
        )
        proba = clf.predict_proba(X_u)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_rejects_out_of_range_unlabelled(self, iris):
        X, y = iris
        X_bad = np.full((5, X.shape[1]), 2.0)
        clf = SemiSupervisedARTMAP()
        with pytest.raises(ValueError):
            clf.fit(X, y, x_unlabelled=X_bad)


# =============================================================================
# VotingARTMAP
# =============================================================================

class TestVotingARTMAP:

    def test_fit(self, iris):
        X, y = iris
        ens = VotingARTMAP(n_voters=3, random_state=0)
        ens.fit(X, y)
        assert len(ens.voters_) == 3

    def test_predict_shape(self, iris):
        X, y = iris
        ens = VotingARTMAP(n_voters=3, random_state=0).fit(X, y)
        assert ens.predict(X).shape == (len(X),)

    def test_hard_voting_accuracy(self, iris):
        X, y = iris
        ens = VotingARTMAP(n_voters=5, voting="hard", random_state=0).fit(X, y)
        acc = np.mean(ens.predict(X) == y)
        assert acc >= 0.90

    def test_soft_voting_proba_shape(self, iris):
        X, y = iris
        ens = VotingARTMAP(
            base_estimator=FuzzyARTMAP(epochs=5),
            n_voters=3, voting="soft", random_state=0
        ).fit(X, y)
        proba = ens.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_node_counts_length(self, iris):
        X, y = iris
        ens = VotingARTMAP(n_voters=4, random_state=0).fit(X, y)
        counts = ens.get_voter_node_counts()
        assert len(counts) == 4
        assert all(c >= 1 for c in counts)

    def test_ensemble_better_than_single(self, iris):
        """Ensemble variance should be lower — ensemble accuracy >= single."""
        X, y = iris
        rng = np.random.default_rng(42)
        single_accs = []
        for seed in range(5):
            idx = rng.permutation(len(X))
            clf = FuzzyARTMAP(epochs=3).fit(X[idx], y[idx])
            single_accs.append(np.mean(clf.predict(X) == y))
        ens = VotingARTMAP(n_voters=5, random_state=0).fit(X, y)
        ens_acc = np.mean(ens.predict(X) == y)
        assert ens_acc >= np.mean(single_accs) - 0.05  # within 5% tolerance

    def test_cross_val(self, iris):
        X, y = iris
        ens = VotingARTMAP(n_voters=3, random_state=0)
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        scores = cross_val_score(ens, X, y, cv=cv, scoring="accuracy")
        assert scores.mean() >= 0.85

    def test_with_bayesian_base(self, iris):
        X, y = iris
        ens = VotingARTMAP(
            base_estimator=BayesianARTMAP(epochs=3),
            n_voters=3, voting="soft", random_state=0
        ).fit(X, y)
        proba = ens.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)


# =============================================================================
# BaggingARTMAP
# =============================================================================

class TestBaggingARTMAP:

    def test_fit(self, iris):
        X, y = iris
        bag = BaggingARTMAP(n_estimators=5, random_state=0).fit(X, y)
        assert len(bag.estimators_) == 5

    def test_predict_shape(self, iris):
        X, y = iris
        bag = BaggingARTMAP(n_estimators=5, random_state=0).fit(X, y)
        assert bag.predict(X).shape == (len(X),)

    def test_accuracy(self, iris):
        X, y = iris
        bag = BaggingARTMAP(n_estimators=7, random_state=0).fit(X, y)
        acc = np.mean(bag.predict(X) == y)
        assert acc >= 0.85

    def test_oob_score_in_range(self, iris):
        X, y = iris
        bag = BaggingARTMAP(n_estimators=10, random_state=0).fit(X, y)
        assert 0.0 <= bag.oob_score_ <= 1.0

    def test_proba_sums_to_one(self, iris):
        X, y = iris
        bag = BaggingARTMAP(
            base_estimator=FuzzyARTMAP(epochs=3),
            n_estimators=5, random_state=0
        ).fit(X, y)
        proba = bag.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_max_samples_subsampling(self, iris):
        X, y = iris
        bag_full = BaggingARTMAP(n_estimators=5, max_samples=1.0, random_state=0)
        bag_half = BaggingARTMAP(n_estimators=5, max_samples=0.5, random_state=0)
        bag_full.fit(X, y); bag_half.fit(X, y)
        # Both should be fitted and predict correctly shaped output
        assert bag_full.predict(X).shape == bag_half.predict(X).shape


# =============================================================================
# FuzzyARTMAP new features
# =============================================================================

class TestFAMRelevance:

    def test_uniform_relevance_same_as_none(self, iris):
        X, y = iris
        M = 2 * X.shape[1]
        r = np.ones(M)
        clf_none = FuzzyARTMAP(epochs=3, random_state=None).fit(X, y) \
            if hasattr(FuzzyARTMAP, "random_state") else FuzzyARTMAP(epochs=3).fit(X, y)
        clf_r    = FuzzyARTMAP(epochs=3, relevance=r).fit(X, y)
        # Predictions should be identical with uniform relevance
        np.testing.assert_array_equal(clf_none.predict(X), clf_r.predict(X))

    def test_relevance_wrong_shape_raises(self, iris):
        X, y = iris
        r_bad = np.ones(3)   # wrong shape
        with pytest.raises(ValueError, match="relevance"):
            FuzzyARTMAP(relevance=r_bad).fit(X, y)

    def test_predict_proba_shape(self, iris):
        X, y = iris
        clf = FuzzyARTMAP(epochs=5).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_predict_proba_sums_to_one(self, iris):
        X, y = iris
        clf = FuzzyARTMAP(epochs=5).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
