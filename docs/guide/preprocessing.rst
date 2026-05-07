Preprocessing Guide
===================

Data Preparation
----------------

Proper data preprocessing is crucial for FuzzyART performance. This guide covers best practices for preparing your data.

Feature Scaling
---------------

FuzzyART models expect inputs in the range [0, 1]. Most implementations handle this automatically, but understanding the process helps optimize results.

Normalization Methods
~~~~~~~~~~~~~~~~~~~~~

**Min-Max Scaling (Recommended)**

Scales features to [0, 1] range:

.. code-block:: python

    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

**Standardization**

Centers features around 0 with unit variance:

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Clip to [0, 1] if needed
    X_scaled = np.clip(X_scaled, 0, 1)

**Custom Scaling**

For domain-specific requirements:

.. code-block:: python

    import numpy as np
    
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)

Feature Engineering
-------------------

Creating Meaningful Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FuzzyART learns from the features you provide. Good feature engineering improves results:

- **Remove irrelevant features**: Features unrelated to your task add noise
- **Combine correlated features**: Reduce redundancy through dimensionality reduction
- **Create domain features**: Transform raw data using domain knowledge

Example:

.. code-block:: python

    # Original features: age, income
    # Create derived feature: age group
    X['age_group'] = pd.cut(X['age'], bins=5)
    
    # Interaction feature
    X['age_income_interaction'] = X['age'] * X['income']

Dimensionality Reduction
~~~~~~~~~~~~~~~~~~~~~~~~

High-dimensional data can be reduced:

.. code-block:: python

    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

Handling Missing Values
-----------------------

FuzzyART does not handle missing values. Remove or impute them first:

**Remove rows with missing values:**

.. code-block:: python

    import pandas as pd
    
    df = pd.read_csv('data.csv')
    df_clean = df.dropna()

**Impute missing values:**

.. code-block:: python

    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
    X_imputed = imputer.fit_transform(X)

**Forward/backward fill (for time series):**

.. code-block:: python

    df = df.fillna(method='ffill')  # forward fill

Handling Categorical Data
--------------------------

Convert categorical features to numeric:

**One-Hot Encoding (for nominal categories):**

.. code-block:: python

    from sklearn.preprocessing import OneHotEncoder
    
    encoder = OneHotEncoder(sparse=False)
    X_encoded = encoder.fit_transform(X[['category']])

**Label Encoding (for ordinal categories):**

.. code-block:: python

    from sklearn.preprocessing import LabelEncoder
    
    encoder = LabelEncoder()
    X['category_encoded'] = encoder.fit_transform(X['category'])

Outlier Detection
-----------------

Outliers can distort clustering:

**Statistical Methods:**

.. code-block:: python

    import numpy as np
    
    # Z-score method
    z_scores = np.abs((X - X.mean()) / X.std())
    outliers = (z_scores > 3).any(axis=1)
    X_clean = X[~outliers]

**IQR Method:**

.. code-block:: python

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    X_clean = X[~outliers]

Class Imbalance (for Supervised Learning)
------------------------------------------

Unbalanced classes can bias learning:

**Resampling:**

.. code-block:: python

    from sklearn.utils import resample
    
    # Oversample minority class
    X_minority = X[y == minority_class]
    X_minority_upsampled = resample(X_minority, n_samples=len(X[y == majority_class]))

**SMOTE (Synthetic Minority Oversampling):**

.. code-block:: python

    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X, y)

Complete Preprocessing Pipeline
--------------------------------

Here's a complete example:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    
    # Load data
    df = pd.read_csv('data.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(df[[col]])
        df = df.drop(col, axis=1)
        df = pd.concat([df, pd.DataFrame(encoded)], axis=1)
    
    # Scale to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Remove outliers (optional)
    z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))
    X_clean = X_scaled[~(z_scores > 3).any(axis=1)]

Best Practices
--------------

1. **Understand your data**: Explore distributions, missing values, and outliers
2. **Document transformations**: Keep track of all preprocessing steps
3. **Fit on training data only**: Avoid data leakage by fitting scalers/encoders only on training set
4. **Preserve test data integrity**: Apply the same transformations to test data
5. **Validate preprocessing**: Visualize results to ensure correctness

.. code-block:: python

    # Good practice: Fit on training data, apply to test data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use transform, not fit_transform
