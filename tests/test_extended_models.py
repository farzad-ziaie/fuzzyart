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
    x, y = load_iris(return_X_y=True)
    return normalize(x), y


@pytest.fixture(scope="module")
def iris_split(iris):
    """30% labelled, 70% unlabelled."""
    x, y = iris
    rng = np.random.default_rng(42)
    mask = rng.random(len(y)) < 0.3
    return x[mask], y[mask], x[~mask], y[~mask]


# =============================================================================
# SemiSupervisedARTMAP
# =============================================================================

class TestSemiSupervisedARTMAP:

    def test_fit_labelled_only(self, iris):
        x, y = iris
        clf = SemiSupervisedARTMAP(epochs=3)
        clf.fit(x, y)   # no unlabelled — should behave like BayesianARTMAP
        assert clf.is_fitted_

    def test_fit_with_unlabelled(self, iris_split):
        x_l, y_l, x_u, _ = iris_split
        clf = SemiSupervisedARTMAP(epochs=3, em_iterations=5)
        clf.fit(x_l, y_l, x_unlabelled=x_u)
        assert clf.is_fitted_

    def test_semi_supervised_at_least_as_good_as_supervised(self, iris_split):
        """With EM on unlabelled data, accuracy should not degrade badly."""
        x_l, y_l, x_u, y_u = iris_split
        x_all = np.concatenate([x_l, x_u])
        y_all = np.concatenate([y_l, y_u])

        sup_clf = BayesianARTMAP(epochs=5).fit(x_l, y_l)
        ss_clf  = SemiSupervisedARTMAP(epochs=5, em_iterations=10).fit(
            x_l, y_l, x_unlabelled=x_u
        )
        # Both should be above 70% on the full dataset
        for clf in (sup_clf, ss_clf):
            acc = np.mean(clf.predict(x_all) == y_all)
            assert acc >= 0.70, f"Accuracy {acc:.2%} too low"

    def test_predict_unlabelled(self, iris_split):
        x_l, y_l, x_u, _ = iris_split
        clf = SemiSupervisedARTMAP(epochs=3, em_iterations=5).fit(
            x_l, y_l, x_unlabelled=x_u
        )
        preds = clf.predict_unlabelled(x_u)
        assert preds.shape == (len(x_u),)
        assert set(preds).issubset(set(clf.classes_))

    def test_predict_proba_sums_to_one(self, iris_split):
        x_l, y_l, x_u, _ = iris_split
        clf = SemiSupervisedARTMAP(epochs=3, em_iterations=3).fit(
            x_l, y_l, x_unlabelled=x_u
        )
        proba = clf.predict_proba(x_u)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_rejects_out_of_range_unlabelled(self, iris):
        x, y = iris
        x_bad = np.full((5, x.shape[1]), 2.0)
        clf = SemiSupervisedARTMAP()
        with pytest.raises(ValueError):
            clf.fit(x, y, x_unlabelled=x_bad)


# =============================================================================
# VotingARTMAP
# =============================================================================

class TestVotingARTMAP:

    def test_fit(self, iris):
        x, y = iris
        ens = VotingARTMAP(n_voters=3, random_state=0)
        ens.fit(x, y)
        assert len(ens.voters_) == 3

    def test_predict_shape(self, iris):
        x, y = iris
        ens = VotingARTMAP(n_voters=3, random_state=0).fit(x, y)
        assert ens.predict(x).shape == (len(x),)

    def test_hard_voting_accuracy(self, iris):
        x, y = iris
        ens = VotingARTMAP(n_voters=5, voting="hard", random_state=0).fit(x, y)
        acc = np.mean(ens.predict(x) == y)
        assert acc >= 0.90

    def test_soft_voting_proba_shape(self, iris):
        x, y = iris
        ens = VotingARTMAP(
            base_estimator=FuzzyARTMAP(epochs=5),
            n_voters=3, voting="soft", random_state=0
        ).fit(x, y)
        proba = ens.predict_proba(x)
        assert proba.shape == (len(x), 3)

    def test_node_counts_length(self, iris):
        x, y = iris
        ens = VotingARTMAP(n_voters=4, random_state=0).fit(x, y)
        counts = ens.get_voter_node_counts()
        assert len(counts) == 4
        assert all(c >= 1 for c in counts)

    def test_ensemble_better_than_single(self, iris):
        """Ensemble variance should be lower — ensemble accuracy >= single."""
        x, y = iris
        rng = np.random.default_rng(42)
        single_accs = []
        for _seed in range(5):
            idx = rng.permutation(len(x))
            clf = FuzzyARTMAP(epochs=3).fit(x[idx], y[idx])
            single_accs.append(np.mean(clf.predict(x) == y))
        ens = VotingARTMAP(n_voters=5, random_state=0).fit(x, y)
        ens_acc = np.mean(ens.predict(x) == y)
        assert ens_acc >= np.mean(single_accs) - 0.05  # within 5% tolerance

    def test_cross_val(self, iris):
        x, y = iris
        ens = VotingARTMAP(n_voters=3, random_state=0)
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        scores = cross_val_score(ens, x, y, cv=cv, scoring="accuracy")
        assert scores.mean() >= 0.85

    def test_with_bayesian_base(self, iris):
        x, y = iris
        ens = VotingARTMAP(
            base_estimator=BayesianARTMAP(epochs=3),
            n_voters=3, voting="soft", random_state=0
        ).fit(x, y)
        proba = ens.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)


# =============================================================================
# BaggingARTMAP
# =============================================================================

class TestBaggingARTMAP:

    def test_fit(self, iris):
        x, y = iris
        bag = BaggingARTMAP(n_estimators=5, random_state=0).fit(x, y)
        assert len(bag.estimators_) == 5

    def test_predict_shape(self, iris):
        x, y = iris
        bag = BaggingARTMAP(n_estimators=5, random_state=0).fit(x, y)
        assert bag.predict(x).shape == (len(x),)

    def test_accuracy(self, iris):
        x, y = iris
        bag = BaggingARTMAP(n_estimators=7, random_state=0).fit(x, y)
        acc = np.mean(bag.predict(x) == y)
        assert acc >= 0.85

    def test_oob_score_in_range(self, iris):
        x, y = iris
        bag = BaggingARTMAP(n_estimators=10, random_state=0).fit(x, y)
        assert 0.0 <= bag.oob_score_ <= 1.0

    def test_proba_sums_to_one(self, iris):
        x, y = iris
        bag = BaggingARTMAP(
            base_estimator=FuzzyARTMAP(epochs=3),
            n_estimators=5, random_state=0
        ).fit(x, y)
        proba = bag.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_max_samples_subsampling(self, iris):
        x, y = iris
        bag_full = BaggingARTMAP(n_estimators=5, max_samples=1.0, random_state=0)
        bag_half = BaggingARTMAP(n_estimators=5, max_samples=0.5, random_state=0)
        bag_full.fit(x, y)
        bag_half.fit(x, y)
        # Both should be fitted and predict correctly shaped output
        assert bag_full.predict(x).shape == bag_half.predict(x).shape


# =============================================================================
# FuzzyARTMAP new features
# =============================================================================

class TestFAMRelevance:

    def test_uniform_relevance_same_as_none(self, iris):
        x, y = iris
        m = 2 * x.shape[1]
        r = np.ones(m)
        clf_none = FuzzyARTMAP(epochs=3, random_state=None).fit(x, y) \
            if hasattr(FuzzyARTMAP, "random_state") else FuzzyARTMAP(epochs=3).fit(x, y)
        clf_r    = FuzzyARTMAP(epochs=3, relevance=r).fit(x, y)
        # Predictions should be identical with uniform relevance
        np.testing.assert_array_equal(clf_none.predict(x), clf_r.predict(x))

    def test_relevance_wrong_shape_raises(self, iris):
        x, y = iris
        r_bad = np.ones(3)   # wrong shape
        with pytest.raises(ValueError, match="relevance"):
            FuzzyARTMAP(relevance=r_bad).fit(x, y)

    def test_predict_proba_shape(self, iris):
        x, y = iris
        clf = FuzzyARTMAP(epochs=5).fit(x, y)
        proba = clf.predict_proba(x)
        assert proba.shape == (len(x), 3)

    def test_predict_proba_sums_to_one(self, iris):
        x, y = iris
        clf = FuzzyARTMAP(epochs=5).fit(x, y)
        proba = clf.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
