# FuzzyART

[![CI](https://github.com/farzad-ziaie/fuzzyart/actions/workflows/ci.yml/badge.svg)](https://github.com/farzad-ziaie/fuzzyart/actions)
[![codecov](https://codecov.io/gh/farzad-ziaie/fuzzyart/branch/main/graph/badge.svg)](https://codecov.io/gh/farzad-ziaie/fuzzyart)
[![PyPI](https://img.shields.io/pypi/v/fuzzyart)](https://pypi.org/project/fuzzyart/)
[![Python](https://img.shields.io/pypi/pyversions/fuzzyart)](https://pypi.org/project/fuzzyart/)
[![Docs](https://readthedocs.org/projects/fuzzyart/badge/?version=latest)](https://fuzzyart.readthedocs.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)


**FuzzyART** is a Python implementation of the **Fuzzy ARTMAP** supervised
classifier, an incremental classifier that learns without catastrophic 
forgetting and requires no pre-specification of the number of categories.

> Carpenter, G.A. (2003). *Default ARTMAP.*  
> IJCNN 2003. DOI: [10.1109/IJCNN.2003.1223900](https://doi.org/10.1109/IJCNN.2003.1223900)

---

## Highlights

- **Online / incremental learning** — `partial_fit` for true streaming use
- **No forgetting** — committed nodes are never destructively overwritten  
- **sklearn-compatible** — works in `Pipeline`, `GridSearchCV`, `cross_val_score`
- **Fast** — pure NumPy, no compilation step
- **Fully typed** — `py.typed` marker, mypy-clean

---

## Installation

```bash
pip install fuzzyart
```

Or with Poetry:

```bash
poetry add fuzzyart
```

---

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from fuzzyart import FuzzyARTMAP
from fuzzyart.preprocessing import normalize

# 1. Load and normalise to [0, 1]
X, y = load_iris(return_X_y=True)
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Train
clf = FuzzyARTMAP(alpha=0.01, beta=0.5, rho_baseline=0.0, epochs=10)
clf.fit(X_train, y_train)
print(f"Committed nodes: {clf.n_committed_}")

# 3. Predict
print(classification_report(y_test, clf.predict(X_test)))
```

---

## Algorithm Overview

Fuzzy ARTMAP combines two ART modules (ART-a and ART-b) with a map field
that learns associations between input patterns and output class labels.

**Training loop (per sample):**

1. **Complement code** input `a → A = [a, 1−a]` (preserves L1 norm)
2. **Compute activation** `T_j` for all committed nodes
3. **Sort descending** and search for a node `J` that satisfies:
   - *Vigilance criterion*: `||fuzzy_and(A, W_J)||₁ / M ≥ ρ`
   - *Correct prediction*: `W^ab_J == k`
4. If found: **update weights** via slow-learning rule
5. If not found: **commit a new node** with `W_J = A`, `W^ab_J = k`

**Key hyperparameters:**

| Parameter | Effect |
|---|---|
| `alpha` | Signal rule; small → more compression |
| `beta` | Learning rate; `1.0` = fast learning |
| `rho_baseline` | Vigilance; higher → finer categories |
| `epsilon` | Match-tracking; negative allows inconsistent samples |

---

## sklearn Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("normalize", FunctionTransformer(normalize)),
    ("clf", FuzzyARTMAP()),
])

param_grid = {
    "clf__alpha": [0.001, 0.01, 0.1],
    "clf__beta": [0.2, 0.5, 0.9],
    "clf__rho_baseline": [0.0, 0.1, 0.3],
}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_weighted")
gs.fit(X, y)
```

---

## Online / Streaming Learning

```python
clf = FuzzyARTMAP(alpha=0.01, beta=0.5)
for X_batch, y_batch in data_stream:
    X_batch = normalize(X_batch)
    clf.partial_fit(X_batch, y_batch)
```

---

## Persistence

```python
clf.save("model.pkl")
clf2 = FuzzyARTMAP.load("model.pkl")
```

---

## Examples

| Notebook | Description |
|---|---|
| [`01_iris_classification`](examples/01_iris_classification.ipynb) | Basic training, evaluation, vigilance sweep |
| [`02_olivetti_faces`](examples/02_olivetti_faces.ipynb) | High-dimensional benchmark vs XGBoost |
| [`03_sklearn_comparison`](examples/03_sklearn_comparison.ipynb) | Pipelines, GridSearchCV, cross-validation |

---

## Development

```bash
git clone https://github.com/farzad-ziaie/fuzzyart
cd fuzzyart
poetry install --with dev,docs,examples

# Run tests
poetry run pytest

# Full CI pipeline
./scripts/ci.sh

# Build docs
./scripts/ci.sh docs
```

---

## License

GPLv3 — see [LICENSE](LICENSE).
