Changelog
=========

All notable changes to FuzzyART will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Added
~~~~~

- Support for online learning in all models
- Bayesian ARTMAP implementation with uncertainty quantification
- Semi-supervised ARTMAP for mixed labeled/unlabeled data
- Ensemble methods combining multiple ART models
- Comprehensive documentation and examples
- Full scikit-learn compatibility

Changed
~~~~~~~

- Improved numerical stability in vigilance calculations
- Enhanced memory efficiency for large datasets
- Refactored preprocessing utilities for better usability

Fixed
~~~~~

- Resolution of edge cases in match tracking
- Corrected learning rate calculations in specific scenarios

Security
~~~~~~~~

- Updated dependencies to address security vulnerabilities

Version 1.0.0 (Initial Release)
-------------------------------

Added
~~~~~

- Core Fuzzy ART implementation
- Fuzzy ARTMAP for supervised learning
- Complete API documentation
- Basic examples and tutorials
- Scikit-learn compatible interface
- Preprocessing utilities
- Hyperparameter optimization examples

Features
^^^^^^^^

**Core Models:**

- ``FuzzyART``: Unsupervised clustering
- ``FuzzyARTMAP``: Supervised classification
- ``BayesianARTMAP``: Probabilistic variant
- ``SemiSupervisedARTMAP``: Mixed learning
- ``EnsembleART``: Ensemble methods

**Key Capabilities:**

- Online learning with incremental data
- Batch training modes
- Automatic cluster number determination
- Scikit-learn ``fit/predict`` interface
- Comprehensive hyperparameter control

**Documentation:**

- Installation guide
- Quick start tutorial
- Algorithm explanations
- Preprocessing guidelines
- Hyperparameter tuning guide
- Three detailed examples
- API reference

**Testing:**

- Comprehensive unit tests
- Integration tests
- Example reproducibility tests

Version 0.5.0 (Beta) - (Hypothetical)
--------------------------------------

Added
~~~~~

- Initial beta release
- Core Fuzzy ART and ARTMAP implementations
- Basic documentation
- Simple examples

Known Limitations
-----------------

Current Limitations
~~~~~~~~~~~~~~~~~~~

- Single-threaded implementation (no parallelization)
- Limited to CPU computation (no GPU support)
- Vigilance-based adaptation only
- No explicit cost function optimization

Planned Improvements
~~~~~~~~~~~~~~~~~~~~

- GPU acceleration for large-scale problems
- Parallel training on multi-core systems
- Adaptive vigilance without manual tuning
- Cost-optimized learning
- Distributed computing support

Dependency Changes
------------------

Current Dependencies
~~~~~~~~~~~~~~~~~~~~

- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- scipy >= 1.5.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

- matplotlib >= 3.0.0 (for visualization)
- pandas >= 1.0.0 (for data handling)
- pytest >= 6.0.0 (for testing)

Breaking Changes
----------------

There are no breaking changes in the initial releases.

Future compatibility notes will be added as the project evolves.

Migration Guide
---------------

No migrations needed for initial version.

Upgrade Instructions
--------------------

To upgrade from any previous version:

.. code-block:: bash

    pip install --upgrade fuzzyart

Contributors
------------

Thanks to all contributors who have helped improve FuzzyART!

Future Releases
---------------

Planned for Next Major Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- GPU support (CUDA/ROCm)
- Distributed computing
- Additional ART variants
- Performance optimizations
- Extended examples

Community Contributions Welcome
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We welcome contributions! Please see ``CONTRIBUTING.rst`` for guidelines.

Release Schedule
----------------

We aim to release updates approximately every 3 months, with bug fixes released as needed.

For more details, see the GitHub releases page.
