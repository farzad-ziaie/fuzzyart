"""Tests for BayesianARTMAP."""

import pickle

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score

from fuzzyart import BayesianARTMAP
from fuzzyart.preprocessing import normalize


@pytest.fixture(scope="module")
def iris():
    x, y = load_iris(return_X_y=True)
    return normalize(x), y


@pytest.fixture
def small_x():
    return np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])

@pytest.fixture
def small_y():
    return np.array([0, 1, 0, 1])


class TestInit:
    def test_defaults(self):
        clf = BayesianARTMAP()
        assert clf.rho_baseline == 0.0
        assert clf.initial_sigma == 1.0
        assert not clf.is_fitted_

    def test_custom(self):
        clf = BayesianARTMAP(rho_baseline=0.3, initial_sigma=0.5, epochs=3)
        assert clf.rho_baseline == 0.3


class TestValidation:
    def test_rejects_3d(self, small_x, small_y):
        with pytest.raises(ValueError, match="2-D"):
            BayesianARTMAP().fit(small_x[np.newaxis], small_y)

    def test_rejects_out_of_range(self, small_y):
        x_bad = np.array([[1.5, 0.5], [0.1, 0.9], [0.5, 0.5], [0.2, 0.8]])
        with pytest.raises(ValueError):
            BayesianARTMAP().fit(x_bad, small_y)

    def test_predict_before_fit(self, small_x):
        with pytest.raises(RuntimeError):
            BayesianARTMAP().predict(small_x)


class TestFit:
    def test_returns_self(self, small_x, small_y):
        clf = BayesianARTMAP()
        assert clf.fit(small_x, small_y) is clf

    def test_is_fitted(self, small_x, small_y):
        clf = BayesianARTMAP().fit(small_x, small_y)
        assert clf.is_fitted_

    def test_commits_nodes(self, small_x, small_y):
        clf = BayesianARTMAP().fit(small_x, small_y)
        assert clf.n_committed_ >= 1

    def test_classes_complete(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        assert set(clf.classes_) == {0, 1, 2}

    def test_partial_fit(self, small_x, small_y):
        clf = BayesianARTMAP()
        clf.partial_fit(small_x[:2], small_y[:2])
        clf.partial_fit(small_x[2:], small_y[2:])
        assert clf.is_fitted_
        assert clf.n_committed_ >= 1


class TestPredict:
    def test_shape(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        assert clf.predict(x).shape == (len(x),)

    def test_known_classes(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=5).fit(x, y)
        assert set(clf.predict(x)).issubset(set(clf.classes_))

    def test_iris_accuracy(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=5).fit(x, y)
        acc = np.mean(clf.predict(x) == y)
        assert acc >= 0.80, f"Accuracy {acc:.2%} below threshold"


class TestPredictProba:
    def test_shape(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        proba = clf.predict_proba(x)
        assert proba.shape == (len(x), 3)

    def test_sums_to_one(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        proba = clf.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_non_negative(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        assert np.all(clf.predict_proba(x) >= 0)

    def test_consistent_with_predict(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        hard = clf.predict(x)
        soft = clf.predict_proba(x)
        soft_pred = np.array([clf.classes_[np.argmax(soft[i])] for i in range(len(x))])
        np.testing.assert_array_equal(hard, soft_pred)


class TestUncertainty:
    def test_uncertainty_shape(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        unc = clf.predict_uncertainty(x)
        assert unc.shape == (len(x),)

    def test_uncertainty_non_negative(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        assert np.all(clf.predict_uncertainty(x) >= 0)


class TestIntrospection:
    def test_means_shape(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        mu = clf.get_category_means()
        assert mu.shape == (clf.n_committed_, clf.n_features_in_)

    def test_sigmas_shape(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        sigma = clf.get_category_sigmas()
        assert sigma.shape == (clf.n_committed_, clf.n_features_in_)
        assert np.all(sigma > 0)


class TestPersistence:
    def test_pickle_roundtrip(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3).fit(x, y)
        blob = pickle.dumps(clf)
        loaded = pickle.loads(blob)
        np.testing.assert_array_equal(clf.predict(x), loaded.predict(x))


class TestSklearn:
    def test_get_set_params(self):
        clf = BayesianARTMAP(rho_baseline=0.2)
        assert clf.get_params()["rho_baseline"] == 0.2
        clf.set_params(rho_baseline=0.5)
        assert clf.rho_baseline == 0.5

    def test_clone(self):
        from sklearn.base import clone
        clf = BayesianARTMAP(rho_baseline=0.3, epochs=5)
        clf2 = clone(clf)
        assert clf2.rho_baseline == 0.3
        assert not clf2.is_fitted_

    def test_cross_val_score(self, iris):
        x, y = iris
        clf = BayesianARTMAP(epochs=3)
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy")
        assert scores.mean() >= 0.80
