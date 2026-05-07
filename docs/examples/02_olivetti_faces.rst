Olivetti Faces Recognition Example
===================================

This example demonstrates FuzzyART for facial recognition on the Olivetti faces dataset, a more complex real-world scenario.

Overview
--------

The Olivetti dataset contains:

- 400 grayscale face images
- 10 different persons
- 40 images per person (different poses, expressions, lighting)
- 64×64 pixel resolution (flattened to 4096 features)

This example shows how to handle high-dimensional data and evaluate performance on a realistic task.

Complete Code
-------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns
    
    from fuzzyart.models import FuzzyARTMAP
    
    # Load dataset
    print("Loading Olivetti faces dataset...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Dimensionality reduction
    print("\nApplying PCA for dimensionality reduction...")
    pca = PCA(n_components=50, random_state=42)
    X_reduced = pca.fit_transform(X)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Reduced shape: {X_reduced.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.3, random_state=42
    )
    
    # Normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining FuzzyARTMAP...")
    model = FuzzyARTMAP(rho=0.7, alpha=0.1, beta=0.7)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion matrix
    ax = axes[0, 0]
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted Person')
    ax.set_ylabel('True Person')
    ax.set_title('Confusion Matrix')
    
    # 2. Per-class accuracy
    ax = axes[0, 1]
    per_class_acc = np.diag(cm) / cm.sum(axis=1)
    ax.bar(range(len(per_class_acc)), per_class_acc)
    ax.set_xlabel('Person')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Person Accuracy')
    ax.set_ylim([0, 1])
    
    # 3. PCA variance explained
    ax = axes[1, 0]
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(cumsum)
    ax.axhline(y=cumsum[49], color='r', linestyle='--', label=f'50 components: {cumsum[49]:.2%}')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Prediction distribution
    ax = axes[1, 1]
    unique, counts = np.unique(y_pred, return_counts=True)
    ax.bar(unique, counts)
    ax.set_xlabel('Predicted Person')
    ax.set_ylabel('Number of Predictions')
    ax.set_title('Prediction Distribution')
    
    plt.tight_layout()
    plt.show()

Step-by-Step Explanation
------------------------

1. **Load High-Dimensional Data**

   .. code-block:: python

       faces = fetch_olivetti_faces()
       X = faces.data  # Shape: (400, 4096)

   Raw pixel data has 4096 features, too high for efficient processing.

2. **Dimensionality Reduction**

   .. code-block:: python

       pca = PCA(n_components=50)
       X_reduced = pca.fit_transform(X)

   Reduces to 50 principal components while retaining ~98% variance.

3. **Data Split and Normalization**

   .. code-block:: python

       X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3)
       
       scaler = MinMaxScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)

4. **Model Training**

   .. code-block:: python

       model = FuzzyARTMAP(rho=0.7, alpha=0.1, beta=0.7)
       model.fit(X_train_scaled, y_train)

   Slightly higher rho (0.7) for finer-grained facial distinctions.

5. **Evaluation**

   .. code-block:: python

       y_pred = model.predict(X_test_scaled)
       accuracy = accuracy_score(y_test, y_pred)

Expected Results
----------------

With proper hyperparameter tuning, you should achieve **85-95% accuracy** on the test set.

Accuracy may vary based on:

- Number of PCA components
- Hyperparameter selection
- Train/test split randomization

Hyperparameter Optimization
----------------------------

Find optimal parameters for this dataset:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    
    # Test different configurations
    param_grid = {
        'rho': np.arange(0.5, 1.0, 0.1),
        'alpha': [0.0, 0.05, 0.1],
        'beta': np.arange(0.5, 1.0, 0.1),
    }
    
    grid_search = GridSearchCV(
        FuzzyARTMAP(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search (this may take a while)...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.2%}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {test_accuracy:.2%}")

Analyzing Misclassifications
-----------------------------

Understand where the model struggles:

.. code-block:: python

    # Find misclassified samples
    misclassified = y_test != y_pred
    
    if misclassified.sum() > 0:
        print(f"Number of misclassifications: {misclassified.sum()}")
        
        # Which persons are confused?
        confusion_pairs = list(zip(y_test[misclassified], y_pred[misclassified]))
        from collections import Counter
        confusion_count = Counter(confusion_pairs)
        
        print("\nMost common confusions:")
        for (true_person, pred_person), count in confusion_count.most_common(5):
            print(f"  Person {true_person} → Person {pred_person}: {count} times")

Comparing with Other Methods
-----------------------------

Compare FuzzyART with other classification algorithms:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    models = {
        'FuzzyARTMAP': FuzzyARTMAP(rho=0.7),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(kernel='rbf'),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    }
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=5, 
            scoring='accuracy'
        )
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
        }
        print(f"{name}: {scores.mean():.2%} (+/- {scores.std():.2%})")
    
    # Visualize comparison
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Algorithm Comparison on Olivetti Faces')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

Advanced: Feature Importance Analysis
--------------------------------------

Understand which principal components matter most:

.. code-block:: python

    # Get component weights from model
    # (implementation depends on model internals)
    
    # Plot PCA components as faces
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.ravel()):
        component = pca.components_[i].reshape(64, 64)
        ax.imshow(component, cmap='gray')
        ax.set_title(f'Component {i}')
        ax.axis('off')
    plt.suptitle('Top 10 Principal Components as Face Images')
    plt.tight_layout()
    plt.show()

Online Learning on Streaming Data
----------------------------------

Process faces as a stream:

.. code-block:: python

    # Create mini-batches
    batch_size = 20
    n_batches = len(X_train_scaled) // batch_size
    
    model = FuzzyARTMAP(rho=0.7)
    
    accuracies = []
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        
        X_batch = X_train_scaled[start:end]
        y_batch = y_train[start:end]
        
        # Online learning
        model.fit(X_batch, y_batch)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Batch Number')
    plt.ylabel('Test Set Accuracy')
    plt.title('Online Learning Progress')
    plt.grid(True, alpha=0.3)
    plt.show()

Key Takeaways
-------------

- High-dimensional data benefits from dimensionality reduction
- FuzzyART can handle realistic facial recognition tasks
- Model evaluation should include confusion analysis
- Hyperparameter tuning is crucial for good performance
- Comparison with baselines provides context

Next Steps
----------

- Explore :doc:`03_sklearn_comparison` for benchmark comparisons
- Review :doc:`../guide/preprocessing` for advanced preprocessing
- Check :doc:`../guide/hyperparameters` for tuning strategies
- See :doc:`../api/models` for other available models
