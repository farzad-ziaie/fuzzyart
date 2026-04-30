# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2024-01-01

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
