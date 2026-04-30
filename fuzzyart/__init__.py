"""
FuzzyART — Fuzzy ARTMAP classifier for Python.

Quick start
-----------
>>> from sklearn.datasets import load_iris
>>> from fuzzyart import FuzzyARTMAP
>>> from fuzzyart.preprocessing import normalize
>>>
>>> X, y = load_iris(return_X_y=True)
>>> X = normalize(X)
>>> clf = FuzzyARTMAP(alpha=0.01, beta=0.5, epochs=5)
>>> clf.fit(X, y)
>>> clf.predict(X[:3])
array([0, 0, 0])
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fuzzyart")
except PackageNotFoundError:
    __version__ = "0.0.0"

from fuzzyart.models.fam import FuzzyARTMAP
from fuzzyart.preprocessing.transforms import (
    complement_code,
    normalize,
    normalize_and_complement_code,
)

__all__ = [
    "FuzzyARTMAP",
    "normalize",
    "complement_code",
    "normalize_and_complement_code",
    "__version__",
]
