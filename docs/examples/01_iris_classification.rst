Iris Classification Example
===========================

This example demonstrates how to use FuzzyART for classification on the classic Iris dataset.

Overview
--------

The Iris dataset contains 150 samples of iris flowers with 4 features each:

- Sepal length
- Sepal width
- Petal length
- Petal width

Each flower is labeled with one of 3 species:

- Setosa
- Versicolor
- Virginica

Complete Code
-------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns
    
    from fuzzyart.models import FuzzyARTMAP
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalize data to [0, 1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = FuzzyARTMAP(rho=0.6, alpha=0.1, beta=0.7)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualize
    plt.figure(figsize=(10, 4))
    
    # Confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Feature importance (cluster sizes)
    plt.subplot(1, 2, 2)
    cluster_counts = np.bincount(model.labels_, minlength=y.max() + 1)
    plt.bar(target_names, cluster_counts)
    plt.ylabel('Number of Clusters')
    plt.title('Clusters per Class')
    
    plt.tight_layout()
    plt.show()

Step-by-Step Explanation
------------------------

1. **Load Data**

   .. code-block:: python

       iris = load_iris()
       X = iris.data  # Features (150, 4)
       y = iris.target  # Labels (150,)

2. **Split Data**

   .. code-block:: python

       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
       )

   This creates 70% training (105 samples) and 30% test (45 samples) sets.

3. **Normalize Features**

   .. code-block:: python

       scaler = MinMaxScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)

   Scales all features to [0, 1] range, required by FuzzyART.

4. **Create Model**

   .. code-block:: python

       model = FuzzyARTMAP(rho=0.6, alpha=0.1, beta=0.7)

   - **rho=0.6**: Moderate vigilance for balanced clustering
   - **alpha=0.1**: Standard choice parameter
   - **beta=0.7**: Moderate learning rate

5. **Train Model**

   .. code-block:: python

       model.fit(X_train_scaled, y_train)

   Model learns to map inputs to output classes.

6. **Make Predictions**

   .. code-block:: python

       y_pred = model.predict(X_test_scaled)

   Predict class labels for test samples.

7. **Evaluate**

   .. code-block:: python

       accuracy = accuracy_score(y_test, y_pred)
       print(f"Accuracy: {accuracy:.2%}")

Expected Results
----------------

With the configuration above, you should achieve approximately **90-95% accuracy** on the test set.

The exact result depends on:

- Random seed (test/train split)
- Hyperparameter values
- Data normalization

Hyperparameter Tuning
---------------------

To improve accuracy, experiment with different hyperparameters:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'rho': [0.4, 0.6, 0.8],
        'alpha': [0.0, 0.1],
        'beta': [0.5, 0.7, 0.9],
    }
    
    grid_search = GridSearchCV(
        FuzzyARTMAP(),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.2%}")
    
    # Use best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

Cross-Validation
----------------

For more robust evaluation, use k-fold cross-validation:

.. code-block:: python

    from sklearn.model_selection import cross_val_score
    
    model = FuzzyARTMAP(rho=0.6)
    
    # 5-fold cross-validation
    scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=5, 
        scoring='accuracy'
    )
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")

Advanced: Online Learning
--------------------------

FuzzyARTMAP supports online learning, allowing it to learn from streaming data:

.. code-block:: python

    model = FuzzyARTMAP(rho=0.6)
    
    # Train on batches
    batch_size = 30
    for i in range(0, len(X_train_scaled), batch_size):
        X_batch = X_train_scaled[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        model.fit(X_batch, y_batch)  # Incremental learning
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Online learning accuracy: {accuracy:.2%}")

Key Takeaways
-------------

- FuzzyART provides a viable alternative to traditional classifiers
- Hyperparameter tuning is important for performance
- Online learning capability enables streaming data scenarios
- Visualization helps understand learned clusters
- Cross-validation provides robust performance estimates

Next Steps
----------

- See :doc:`../guide/hyperparameters` for detailed tuning strategies
- Check :doc:`02_olivetti_faces` for a more complex example
- Review :doc:`../api/models` for other available models
