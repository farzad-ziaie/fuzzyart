"""Tests for fuzzyart.models.fam.FuzzyARTMAP."""

import pickle
import tempfile

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from fuzzyart import FuzzyARTMAP
from fuzzyart.preprocessing import normalize

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_data():
    x, y = load_iris(return_X_y=True)
    return normalize(x), y


@pytest.fixture
def fitted_clf(iris_data):
    x, y = iris_data
    clf = FuzzyARTMAP(alpha=0.01, beta=0.5, epochs=5)
    clf.fit(x, y)
    return clf


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_params(self):
        clf = FuzzyARTMAP()
        assert clf.alpha == 0.01
        assert clf.beta == 0.2
        assert clf.epsilon == -0.001
        assert clf.rho_baseline == 0.0
        assert clf.epochs == 1

    def test_custom_params(self):
        clf = FuzzyARTMAP(alpha=0.1, beta=0.9, epochs=10)
        assert clf.alpha == 0.1
        assert clf.beta == 0.9
        assert clf.epochs == 10

    def test_not_fitted_flag(self):
        clf = FuzzyARTMAP()
        assert not clf.is_fitted_


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_raises_on_3d_input(self, simple_x, simple_y):
        clf = FuzzyARTMAP()
        with pytest.raises(ValueError, match="2-D"):
            clf.fit(simple_x[np.newaxis], simple_y)

    def test_raises_on_out_of_range(self, simple_y):
        clf = FuzzyARTMAP()
        x_bad = np.array([[1.5, 0.5], [0.1, 0.9], [0.5, 0.5]])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            clf.fit(x_bad, simple_y)

    def test_raises_on_length_mismatch(self, simple_x):
        clf = FuzzyARTMAP()
        with pytest.raises(ValueError, match="same number of samples"):
            clf.fit(simple_x, np.array([0, 1]))

    def test_predict_before_fit_raises(self, simple_x):
        clf = FuzzyARTMAP()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(simple_x)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestFit:
    def test_returns_self(self, simple_x, simple_y):
        clf = FuzzyARTMAP()
        result = clf.fit(simple_x, simple_y)
        assert result is clf

    def test_is_fitted_after_fit(self, simple_x, simple_y):
        clf = FuzzyARTMAP()
        clf.fit(simple_x, simple_y)
        assert clf.is_fitted_

    def test_commits_at_least_one_node(self, simple_x, simple_y):
        clf = FuzzyARTMAP()
        clf.fit(simple_x, simple_y)
        assert clf.n_committed_ >= 1

    def test_classes_detected(self, iris_data):
        x, y = iris_data
        clf = FuzzyARTMAP().fit(x, y)
        assert set(clf.classes_) == {0, 1, 2}

    def test_multiple_epochs_converge(self, iris_data):
        """After enough epochs the node count stabilises (stops growing)."""
        x, y = iris_data
        clf4 = FuzzyARTMAP(epochs=4).fit(x, y)
        clf5 = FuzzyARTMAP(epochs=5).fit(x, y)
        # Node count difference between epoch 4 and 5 should be small
        diff = abs(clf5.n_committed_ - clf4.n_committed_)
        assert diff <= max(5, 0.05 * clf4.n_committed_), \
            f"Node count still changing significantly at epoch 5: {clf4.n_committed_} → {clf5.n_committed_}"

    def test_higher_vigilance_more_nodes(self, iris_data):
        """Higher rho_baseline → finer categories → more nodes."""
        x, y = iris_data
        clf_low = FuzzyARTMAP(rho_baseline=0.0, epochs=3).fit(x, y)
        clf_high = FuzzyARTMAP(rho_baseline=0.5, epochs=3).fit(x, y)
        assert clf_high.n_committed_ >= clf_low.n_committed_


class TestPredict:
    def test_output_shape(self, fitted_clf, iris_data):
        x, _ = iris_data
        preds = fitted_clf.predict(x)
        assert preds.shape == (x.shape[0],)

    def test_predictions_are_known_classes(self, fitted_clf, iris_data):
        x, _ = iris_data
        preds = fitted_clf.predict(x)
        assert set(preds).issubset(set(fitted_clf.classes_))

    def test_iris_accuracy_above_threshold(self, iris_data):
        """FuzzyARTMAP should achieve > 90% on Iris with 5 epochs."""
        x, y = iris_data
        clf = FuzzyARTMAP(alpha=0.01, beta=0.5, epochs=5).fit(x, y)
        acc = np.mean(clf.predict(x) == y)
        assert acc >= 0.90, f"Accuracy {acc:.2%} below threshold."


class TestPartialFit:
    def test_incremental_same_result(self, simple_x, simple_y):
        clf1 = FuzzyARTMAP().fit(simple_x, simple_y)

        clf2 = FuzzyARTMAP()
        for xi, yi in zip(simple_x, simple_y, strict=True):
            clf2.partial_fit(xi[np.newaxis], np.array([yi]))

        np.testing.assert_array_equal(
            clf1.predict(simple_x), clf2.predict(simple_x)
        )

    def test_partial_fit_is_fitted(self, simple_x, simple_y):
        clf = FuzzyARTMAP()
        clf.partial_fit(simple_x[:1], simple_y[:1])
        assert clf.is_fitted_


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_get_node_weights_shape(self, fitted_clf):
        w = fitted_clf.get_node_weights()
        assert w.shape == (fitted_clf.n_committed_, 2 * fitted_clf.n_features_in_)

    def test_get_node_labels_length(self, fitted_clf):
        labels = fitted_clf.get_node_labels()
        assert len(labels) == fitted_clf.n_committed_

    def test_summary_keys(self, fitted_clf):
        s = fitted_clf.summary()
        for key in ("n_committed", "n_features", "n_classes", "classes", "hyperparameters"):
            assert key in s


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, fitted_clf, iris_data):
        x, _ = iris_data
        preds_before = fitted_clf.predict(x)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        fitted_clf.save(path)
        loaded = FuzzyARTMAP.load(path)
        preds_after = loaded.predict(x)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_pickle_compatible(self, fitted_clf, iris_data):
        x, _ = iris_data
        blob = pickle.dumps(fitted_clf)
        loaded = pickle.loads(blob)
        np.testing.assert_array_equal(
            fitted_clf.predict(x), loaded.predict(x)
        )


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------

class TestSklearnCompat:
    def test_get_params(self):
        clf = FuzzyARTMAP(alpha=0.05, beta=0.3)
        params = clf.get_params()
        assert params["alpha"] == 0.05
        assert params["beta"] == 0.3

    def test_set_params(self):
        clf = FuzzyARTMAP()
        clf.set_params(alpha=0.99)
        assert clf.alpha == 0.99

    def test_clone(self):
        from sklearn.base import clone
        clf = FuzzyARTMAP(alpha=0.05, epochs=3)
        clf2 = clone(clf)
        assert clf2.alpha == clf.alpha
        assert not clf2.is_fitted_

    def test_cross_val_score(self, iris_data):
        from sklearn.model_selection import StratifiedKFold
        x, y = iris_data
        clf = FuzzyARTMAP(alpha=0.01, beta=0.5, epochs=3)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, x, y, cv=cv, scoring="f1_weighted")
        assert scores.mean() >= 0.85
