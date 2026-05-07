FuzzyART Documentation
======================

Welcome to FuzzyART, a Python implementation of Fuzzy Adaptive Resonance Theory (ART) and related neural network models for clustering and classification.

FuzzyART provides efficient, interpretable machine learning algorithms with online learning capabilities. This documentation will help you get started, understand the algorithms, and build powerful applications.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   readme
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/models
   api/preprocessing
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: About

   changelog
   contributing

Key Features
============

- **Online Learning**: Train models incrementally with streaming data
- **Automatic Cluster Discovery**: No need to specify the number of clusters
- **Interpretable Prototypes**: Understand what your model has learned
- **Scikit-learn Compatible**: Use familiar ``fit/predict`` interface
- **Multiple Variants**: ART, ARTMAP, Bayesian, Semi-supervised, Ensemble
- **Fast Inference**: Efficient prediction on new data

Quick Links
===========

- **New to FuzzyART?** Start with :doc:`quickstart`
- **Want to install?** See :doc:`installation`
- **Need an example?** Check :doc:`examples/index`
- **Curious about algorithms?** Read :doc:`guide/algorithm`
- **Want to contribute?** Visit :doc:`contributing`

Documentation Sections
======================

**Getting Started**

Begin your journey with FuzzyART by learning about the project, installing it, and running your first example.

- :doc:`readme` - Project overview and features
- :doc:`installation` - Installation instructions (PyPI, source, Poetry, development)
- :doc:`quickstart` - Basic example to get started in minutes

**User Guide**

Dive deeper with comprehensive guides on algorithms, data preprocessing, and hyperparameter tuning.

- :doc:`guide/algorithm` - Understand Adaptive Resonance Theory
- :doc:`guide/preprocessing` - Prepare your data properly
- :doc:`guide/hyperparameters` - Tune parameters for better results

**API Reference**

Complete reference documentation for all modules and functions.

- :doc:`api/models` - Model classes and methods
- :doc:`api/preprocessing` - Data preprocessing utilities
- :doc:`api/utils` - Utility functions

**Examples**

Learn through practical, working examples on real datasets.

- :doc:`examples/01_iris_classification` - Basic classification
- :doc:`examples/02_olivetti_faces` - Advanced high-dimensional data
- :doc:`examples/03_sklearn_comparison` - Compare with other algorithms

**About**

Project information and contribution guidelines.

- :doc:`changelog` - Version history
- :doc:`contributing` - How to contribute to the project

Models Available
================

FuzzyART provides several neural network models:

**Unsupervised Learning**

- **Fuzzy ART**: Clustering without labels

**Supervised Learning**

- **Fuzzy ARTMAP**: Standard classification
- **Bayesian ARTMAP**: Probabilistic variant with uncertainty
- **Semi-supervised ARTMAP**: Learn from mixed labeled/unlabeled data

**Ensemble Methods**

- **Ensemble ART**: Combine multiple models

Basic Usage
===========

Here's a simple example to get started:

.. code-block:: python

    from fuzzyart.models import FuzzyART
    import numpy as np
    
    # Generate sample data
    X = np.random.rand(100, 2)
    
    # Create and fit model
    model = FuzzyART(rho=0.5)
    model.fit(X)
    
    # Get cluster labels
    labels = model.predict(X)
    print(f"Number of clusters: {len(np.unique(labels))}")

For supervised learning:

.. code-block:: python

    from fuzzyart.models import FuzzyARTMAP
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3
    )
    
    # Train model
    model = FuzzyARTMAP(rho=0.6)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2%}")

See :doc:`quickstart` for more examples.

Common Tasks
============

**Install FuzzyART**

See :doc:`installation` for detailed instructions including pip and Poetry options.

.. code-block:: bash

    pip install fuzzyart

Or with Poetry:

.. code-block:: bash

    poetry add fuzzyart

**Learn the Basics**

Follow :doc:`quickstart` for your first classification task.

**Understand Algorithms**

Read :doc:`guide/algorithm` for the theory behind Adaptive Resonance Theory.

**Prepare Your Data**

Check :doc:`guide/preprocessing` for data preparation best practices.

**Tune Hyperparameters**

Follow :doc:`guide/hyperparameters` for systematic parameter optimization.

**See Real Examples**

Explore :doc:`examples/01_iris_classification`, :doc:`examples/02_olivetti_faces`, and :doc:`examples/03_sklearn_comparison`.

Requirements
============

- Python 3.8 or higher
- NumPy
- Scikit-learn
- SciPy

Optional dependencies:

- Matplotlib (for visualization)
- Pandas (for data handling)

See :doc:`installation` for detailed setup instructions.

Support & Community
===================

- **Documentation**: You're reading it!
- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions and share ideas
- **Contributing**: See :doc:`contributing` to get involved

Author & Website
================

**Author**: Farzad Ziaie Nezhad

**Website**: https://farzad-ziaie.github.io/

Getting Help
============

If you encounter issues:

1. Check the :doc:`quickstart` guide
2. Review relevant examples in :doc:`examples/index`
3. Search the API documentation in :doc:`api/models`
4. Check the :doc:`guide/preprocessing` for data preparation issues
5. Read :doc:`guide/hyperparameters` for performance problems
6. Open an issue on GitHub with reproducible code

Citation
========

If you use FuzzyART in research, please cite this project. Citation information will be updated.

License
=======

FuzzyART is provided under an open-source license. See the repository for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::

   This documentation is for FuzzyART. For more information, visit the
   GitHub repository.

.. warning::

   This is the documentation for FuzzyART. Make sure you have installed it
   correctly before running the examples. See :doc:`installation` for details.

Version Information
===================

- **Current Version**: 1.0.0
- **Last Updated**: 2024
- **Python Version**: 3.8+
- **Status**: Production Ready
- **Author**: Farzad Ziaie Nezhad

Next Steps
==========

Ready to get started?

1. **First time here?** → Start with :doc:`readme`
2. **Need to install?** → Follow :doc:`installation`
3. **Want a quick example?** → Check :doc:`quickstart`
4. **Looking for detailed guides?** → See :doc:`guide/index`
5. **Want to see real examples?** → Explore :doc:`examples/index`
6. **Need API reference?** → Visit the API section above
