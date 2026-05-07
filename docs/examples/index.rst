Examples
========

Practical examples demonstrating FuzzyART capabilities on real datasets.

These examples show how to use FuzzyART for different types of problems and datasets.

.. toctree::
   :maxdepth: 2

   01_iris_classification
   02_olivetti_faces
   03_sklearn_comparison

Example 1: Iris Classification
-------------------------------

A beginner-friendly introduction to FuzzyART using the classic Iris dataset.

**Topics covered:**

- Loading and preparing data
- Normalizing features
- Training a classifier
- Evaluating performance
- Hyperparameter tuning
- Cross-validation

**Dataset:** 150 flower samples, 4 features, 3 classes

**Difficulty:** Beginner

:doc:`01_iris_classification` →

Example 2: Olivetti Faces Recognition
--------------------------------------

An advanced example with high-dimensional facial image data.

**Topics covered:**

- Working with high-dimensional data
- Dimensionality reduction (PCA)
- Complex evaluation metrics
- Confusion matrix interpretation
- Comparing with other algorithms
- Online learning on streams

**Dataset:** 400 face images, 64×64 pixels, 10 people

**Difficulty:** Intermediate/Advanced

:doc:`02_olivetti_faces` →

Example 3: Algorithm Comparison
--------------------------------

Comprehensive benchmark comparing FuzzyART with popular scikit-learn algorithms.

**Topics covered:**

- Multi-dataset evaluation
- Algorithm comparison on multiple datasets
- Performance metrics
- Computational efficiency analysis
- Hyperparameter sensitivity analysis
- Algorithm selection recommendations

**Datasets:** Iris, Wine, Breast Cancer (3 different datasets)

**Algorithms:** FuzzyART, KNN, SVM, Random Forest, Logistic Regression, Gradient Boosting

**Difficulty:** Intermediate/Advanced

:doc:`03_sklearn_comparison` →

Quick Navigation
================

**I'm new to FuzzyART**

→ Start with :doc:`01_iris_classification`

**I have complex data**

→ Move to :doc:`02_olivetti_faces`

**I want to see how it compares**

→ Check :doc:`03_sklearn_comparison`

Learning Path
=============

**Beginner Path (30 minutes)**

1. Read :doc:`01_iris_classification` completely
2. Run the code examples
3. Try changing hyperparameters
4. Observe how results change

**Intermediate Path (1-2 hours)**

1. Work through Example 1
2. Explore Example 2 with focus on:
   - Dimensionality reduction
   - PCA visualization
   - Confusion matrix analysis
3. Run hyperparameter tuning

**Advanced Path (2-4 hours)**

1. Complete all three examples
2. Modify examples for your own data
3. Compare results across examples
4. Implement your own variations
5. Read :doc:`../guide/index` for deeper understanding

Dataset Overview
================

**Iris Dataset (Example 1)**

- Samples: 150
- Features: 4
- Classes: 3
- Data type: Numerical (continuous)
- Task: Classification
- Difficulty: Easy (well-separated classes)

**Olivetti Faces (Example 2)**

- Samples: 400
- Features: 4,096 (images, reduced to ~50 with PCA)
- Classes: 10 (persons)
- Data type: Images (grayscale)
- Task: Recognition/Classification
- Difficulty: Medium (high-dimensional)

**Comparison Datasets (Example 3)**

- Wine: 178 samples, 13 features, 3 classes
- Breast Cancer: 569 samples, 30 features, 2 classes
- Used for comparative analysis across algorithms

Code Examples by Topic
======================

**Data Loading & Preparation**

All examples show proper data loading, splitting, and normalization.

**Model Training**

Each example demonstrates fitting a model:

- Example 1: Basic fit/predict
- Example 2: Online learning on batches
- Example 3: Cross-validation training

**Evaluation & Visualization**

Examples show different evaluation approaches:

- Example 1: Confusion matrix
- Example 2: Per-class metrics
- Example 3: Multi-algorithm comparison

**Hyperparameter Tuning**

All examples include parameter optimization:

- Example 1: Manual tuning
- Example 2: Grid search
- Example 3: Sensitivity analysis

Common Tasks Demonstrated
==========================

**Classification on Standard Data**

→ See Example 1

**High-Dimensional Data Handling**

→ See Example 2

**Algorithm Performance Comparison**

→ See Example 3

**Hyperparameter Optimization**

→ All examples

**Cross-Validation**

→ Examples 1 and 3

**Batch/Online Learning**

→ Example 2

**Confusion Matrix Analysis**

→ Examples 1 and 2

**Computational Efficiency**

→ Example 3

**Feature Importance**

→ Example 2

Tips for Running Examples
=========================

1. **Install dependencies**: Ensure matplotlib, pandas, numpy are installed
2. **Run in order**: Start with Example 1, progress to others
3. **Modify code**: Change parameters to see results
4. **Visualize results**: Use plots to understand behavior
5. **Compare metrics**: Look at multiple evaluation metrics
6. **Timing**: Note execution times for performance analysis

Expected Results
================

**Example 1 (Iris)**

- Accuracy: 85-95%
- Number of clusters/categories created
- Confusion matrix showing class-wise performance

**Example 2 (Olivetti Faces)**

- Accuracy: 75-90%
- Per-person recognition rates
- Confusion patterns between similar faces

**Example 3 (Comparison)**

- FuzzyART typically achieves competitive accuracy
- Different strengths across datasets
- Clear computational efficiency advantages

Troubleshooting Examples
========================

**ImportError: No module named 'fuzzyart'**

→ Install: ``pip install fuzzyart``

**Data loading errors**

→ Check internet connection (automatic downloads)

**Memory errors on Example 2**

→ Reduce dataset size or increase PCA components threshold

**Long training times**

→ Use smaller subset or reduce cross-validation folds

**Different results each run**

→ This is normal due to random initialization. Set random_state for reproducibility.

Next Steps After Examples
==========================

After completing the examples:

1. **Learn more theory**: Read :doc:`../guide/algorithm`
2. **Improve preprocessing**: Study :doc:`../guide/preprocessing`
3. **Optimize further**: Explore :doc:`../guide/hyperparameters`
4. **Apply to your data**: Use examples as templates
5. **Contribute**: See :doc:`../contributing`

Resources
=========

- :doc:`../api/models` - Model API reference
- :doc:`../guide/index` - Detailed guides
- :doc:`../installation` - Installation help (including Poetry)
- :doc:`../contributing` - Contribution guidelines

Code Availability
=================

All example code is:

- ✅ Ready to run
- ✅ Fully commented
- ✅ Reproducible (with set random seeds)
- ✅ Using public datasets
- ✅ Compatible with latest versions

Questions or Issues?
====================

If you encounter problems:

1. Check the example comments
2. Review related guide section
3. Check API documentation
4. Search GitHub issues
5. Open a new issue with reproducible code

Happy learning!
