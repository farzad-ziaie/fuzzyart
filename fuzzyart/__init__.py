"""
FuzzyART — ART-family classifiers for Python.
"""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("fuzzyart")
except PackageNotFoundError:
    __version__ = "0.2.0"

from fuzzyart.models.fam import FuzzyARTMAP
from fuzzyart.models.bayesian_artmap import BayesianARTMAP
from fuzzyart.models.semisupervised_artmap import SemiSupervisedARTMAP
from fuzzyart.models.ensemble import VotingARTMAP, BaggingARTMAP
from fuzzyart.preprocessing.transforms import (
    complement_code,
    normalize,
    normalize_and_complement_code,
)

__all__ = [
    "FuzzyARTMAP",
    "BayesianARTMAP",
    "SemiSupervisedARTMAP",
    "VotingARTMAP",
    "BaggingARTMAP",
    "normalize",
    "complement_code",
    "normalize_and_complement_code",
    "__version__",
]
