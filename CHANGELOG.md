# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2020-01-01

### Added
- `FuzzyARTMAP` classifier with full sklearn compatibility
- `partial_fit` for online / streaming learning
- `save` / `load` persistence helpers
- `fuzzyart.preprocessing`: `normalize`, `complement_code`, `truncated_svd`
- `fuzzyart.utils.math`: `l1_norm`, `fuzzy_and`, `complement`
- Full pytest suite with > 80% coverage
- Sphinx documentation with ReadTheDocs config
- Example notebooks: Iris, Olivetti faces, sklearn integration
- GitHub Actions CI/CD pipeline
- Poetry packaging, PyPI-ready

## [0.2.0] — 2026-04-31

### Added
- `BayesianARTMAP` — diagonal Gaussian categories, calibrated `predict_proba`,
  `predict_uncertainty` (Shannon entropy), Mahalanobis-based vigilance,
  Welford online variance update (Vigdor & Lerner 2007)
- `SemiSupervisedARTMAP` — EM-based semi-supervised extension; exploits
  unlabelled data via soft E-step + M-step weight updates
- `VotingARTMAP` — ensemble over shuffled orderings; hard + soft voting;
  compatible with any base estimator
- `BaggingARTMAP` — bootstrap-aggregated ensemble with OOB accuracy estimate
- `FuzzyARTMAP.relevance` — per-feature relevance weighting (Andonie & Sasu 2006)
- `FuzzyARTMAP.predict_proba` — softmax over per-class activation sums
- Vectorised signal computation in `FuzzyARTMAP` (NumPy stack + broadcast)
- Example notebook `04_all_models_medical_benchmark.ipynb`
- 49 additional tests; total 95 tests, 94% coverage
