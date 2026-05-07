Hyperparameter Tuning Guide
===========================

Overview
--------

FuzzyART models have several hyperparameters that control learning behavior. Tuning these parameters is essential for achieving good performance on your specific dataset.

Core Hyperparameters
--------------------

Vigilance Parameter (rho)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Controls the sensitivity of cluster matching (range: 0 to 1)

- **Low values (0.0-0.3)**: Creates coarse, broad clusters
- **Medium values (0.4-0.6)**: Balanced clustering
- **High values (0.7-1.0)**: Creates fine-grained, specific clusters

**Impact on clustering:**

- Too low: Too few clusters, information loss
- Too high: Too many clusters, fragmentation

**Guidelines:**

.. code-block:: python

    # Conservative clustering
    model = FuzzyART(rho=0.3)
    
    # Balanced approach
    model = FuzzyART(rho=0.5)
    
    # Fine-grained clustering
    model = FuzzyART(rho=0.8)

**How to tune:**

Start with rho=0.5 and adjust based on results:

.. code-block:: python

    for rho in [0.3, 0.5, 0.7, 0.9]:
        model = FuzzyART(rho=rho)
        model.fit(X)
        print(f"rho={rho}: {len(np.unique(model.labels_))} clusters")

Learning Rate (beta)
~~~~~~~~~~~~~~~~~~~~

**Description**: Controls how much weights are updated during learning (range: 0 to 1)

- **Low values (0.1-0.3)**: Slow, conservative learning
- **Medium values (0.5-0.7)**: Standard learning
- **High values (0.9-1.0)**: Fast, aggressive learning

**Impact:**

- Too low: Slow convergence, weights change minimally
- Too high: Can oscillate, unstable learning

**Guidelines:**

.. code-block:: python

    # Conservative learning
    model = FuzzyARTMAP(beta=0.1)
    
    # Standard learning
    model = FuzzyARTMAP(beta=0.7)
    
    # Fast learning
    model = FuzzyARTMAP(beta=1.0)

Choice Parameter (alpha)
~~~~~~~~~~~~~~~~~~~~~~~~

**Description**: Controls contribution of cluster weight in choice function (range: typically 0 to 0.1)

- **Low values (0.0-0.01)**: Weight magnitudes dominate
- **Higher values (0.05-0.1)**: Balances size and overlap

**Guidelines:**

.. code-block:: python

    model = FuzzyARTMAP(alpha=0.0)   # Simple choice
    model = FuzzyARTMAP(alpha=0.01)  # Standard
    model = FuzzyARTMAP(alpha=0.1)   # Complex scenarios

Tuning Strategies
-----------------

Grid Search
~~~~~~~~~~~

Test combinations of hyperparameters:

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from fuzzyart.models import FuzzyART
    
    param_grid = {
        'rho': [0.3, 0.5, 0.7, 0.9],
        'alpha': [0.0, 0.01, 0.1],
    }
    
    model = FuzzyART()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='silhouette')
    grid_search.fit(X)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

Random Search
~~~~~~~~~~~~~

Sample random parameter combinations:

.. code-block:: python

    from sklearn.model_selection import RandomizedSearchCV
    
    param_dist = {
        'rho': np.random.uniform(0.1, 0.95, 10),
        'alpha': np.random.uniform(0.0, 0.1, 10),
        'beta': np.random.uniform(0.1, 1.0, 10),
    }
    
    random_search = RandomizedSearchCV(
        FuzzyARTMAP(), param_dist, n_iter=20, cv=5
    )
    random_search.fit(X_train, y_train)

Manual Tuning
~~~~~~~~~~~~~

Iterative refinement based on validation metrics:

.. code-block:: python

    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    best_score = -np.inf
    best_params = {}
    
    for rho in np.arange(0.1, 1.0, 0.1):
        model = FuzzyART(rho=rho)
        labels = model.fit_predict(X)
        
        # Use silhouette score
        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_params = {'rho': rho}
    
    print(f"Best rho: {best_params['rho']} (score: {best_score:.3f})")

Validation Metrics
------------------

For Unsupervised Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Silhouette Coefficient** (higher is better):

.. code-block:: python

    from sklearn.metrics import silhouette_score
    
    score = silhouette_score(X, labels)
    # Range: -1 to 1, where 1 is best

**Davies-Bouldin Index** (lower is better):

.. code-block:: python

    from sklearn.metrics import davies_bouldin_score
    
    score = davies_bouldin_score(X, labels)
    # Lower values indicate better clustering

**Calinski-Harabasz Index** (higher is better):

.. code-block:: python

    from sklearn.metrics import calinski_harabasz_score
    
    score = calinski_harabasz_score(X, labels)
    # Higher values indicate better separation

For Supervised Learning
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.metrics import accuracy_score, f1_score, precision_score
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

Hyperparameter Interaction Effects
-----------------------------------

Parameters don't act independently. Consider interactions:

**rho and beta interaction:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    rho_values = np.arange(0.1, 1.0, 0.1)
    beta_values = np.arange(0.1, 1.1, 0.1)
    scores = np.zeros((len(rho_values), len(beta_values)))
    
    for i, rho in enumerate(rho_values):
        for j, beta in enumerate(beta_values):
            model = FuzzyARTMAP(rho=rho, beta=beta)
            model.fit(X_train, y_train)
            scores[i, j] = model.score(X_test, y_test)
    
    plt.imshow(scores, aspect='auto', cmap='viridis')
    plt.xlabel('beta')
    plt.ylabel('rho')
    plt.colorbar(label='Accuracy')
    plt.show()

Dataset-Specific Recommendations
---------------------------------

Small Datasets (< 1000 samples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use higher rho (0.6-0.9) to avoid overfragmentation
- Lower beta (0.1-0.3) for stable learning
- Perform k-fold cross-validation (k=5 or 10)

Large Datasets (> 10000 samples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Can use lower rho (0.3-0.5) safely
- Higher beta (0.7-1.0) for faster learning
- Use random subsampling for parameter search

High-Dimensional Data (> 100 features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Consider dimensionality reduction first
- Start with lower rho (0.2-0.4) to avoid many clusters
- May need careful feature scaling

Best Practices
--------------

1. **Start simple**: Begin with default parameters, adjust one at a time
2. **Use validation sets**: Don't tune on test data
3. **Visualize results**: Plot metrics vs. hyperparameters
4. **Document choices**: Record why you selected specific values
5. **Cross-validate**: Use k-fold CV for robust estimates
6. **Consider computational cost**: Some parameter combinations may be slow

Complete Tuning Example
-----------------------

.. code-block:: python

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, silhouette_score
    import numpy as np
    
    # Define scoring function
    silhouette_scorer = make_scorer(silhouette_score)
    
    # Test different configurations
    configs = [
        {'rho': 0.3, 'alpha': 0.0, 'beta': 0.1},
        {'rho': 0.5, 'alpha': 0.01, 'beta': 0.5},
        {'rho': 0.7, 'alpha': 0.1, 'beta': 0.9},
    ]
    
    results = []
    for config in configs:
        model = FuzzyARTMAP(**config)
        scores = cross_val_score(
            model, X, y, cv=5, 
            scoring='accuracy'
        )
        results.append({
            'config': config,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        })
        print(f"Config: {config}")
        print(f"  Mean: {scores.mean():.3f} ± {scores.std():.3f}\n")
    
    # Select best
    best = max(results, key=lambda x: x['mean_score'])
    print(f"Best configuration: {best['config']}")
