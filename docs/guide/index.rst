User Guide
==========

Comprehensive guides for understanding and using FuzzyART.

This section provides in-depth documentation on the algorithms, data preparation, and hyperparameter tuning.

.. toctree::
   :maxdepth: 2

   algorithm
   preprocessing
   hyperparameters

Algorithm Guide
---------------

Learn the theory behind Adaptive Resonance Theory and how FuzzyART models work.

- Understand ART fundamentals
- Explore Fuzzy ART components
- Learn about vigilance and choice functions
- Discover learning mechanisms
- Compare different model variants

:doc:`algorithm` →

Preprocessing Guide
-------------------

Prepare your data properly for optimal model performance.

- Feature scaling and normalization
- Feature engineering techniques
- Handling missing values
- Encoding categorical data
- Outlier detection and handling
- Addressing class imbalance

:doc:`preprocessing` →

Hyperparameter Tuning Guide
---------------------------

Learn how to find optimal hyperparameters for your specific problem.

- Understand each hyperparameter
- Implement tuning strategies (grid search, random search, manual)
- Use validation metrics
- Analyze parameter interactions
- Get dataset-specific recommendations

:doc:`hyperparameters` →

Learning Path
=============

**Beginner:**

1. Start with :doc:`algorithm` to understand the concepts
2. Move to :doc:`preprocessing` to prepare your data
3. Then use :doc:`hyperparameters` to optimize your model

**Intermediate:**

1. Review the algorithm guide for deeper understanding
2. Implement advanced preprocessing techniques
3. Systematically tune hyperparameters

**Advanced:**

1. Understand mathematical details in algorithm guide
2. Implement custom preprocessing pipelines
3. Perform multi-dimensional hyperparameter optimization

Quick Reference
===============

**Algorithm Topics**

- Adaptive Resonance Theory (ART) basics
- Fuzzy ART components
- Fuzzy ARTMAP (supervised learning)
- Bayesian ARTMAP (probabilistic)
- Semi-supervised variants
- Ensemble methods

**Preprocessing Topics**

- Min-max scaling
- Standardization
- Feature engineering
- PCA dimensionality reduction
- Missing value imputation
- Categorical encoding
- Outlier detection
- Class balancing

**Hyperparameter Topics**

- Vigilance parameter (rho)
- Learning rate (beta)
- Choice parameter (alpha)
- Grid search
- Random search
- Cross-validation
- Validation metrics

Getting Started with This Guide
================================

Choose your topic based on your needs:

**I want to understand how FuzzyART works**

→ Read :doc:`algorithm` for complete explanation

**I have data that needs preparation**

→ Check :doc:`preprocessing` for techniques and examples

**My model isn't performing well**

→ See :doc:`hyperparameters` for tuning strategies

Tips for Using This Guide
==========================

- **Code examples**: All code is ready to run
- **Tables**: Use for quick reference
- **Cross-references**: Links to related topics
- **Diagrams**: Mathematical notation and visual representations
- **Best practices**: Recommendations based on experience

Additional Resources
====================

For API reference, see the main documentation.

For working examples, visit the :doc:`../examples/index` section.

For contribution information, see :doc:`../contributing`.
