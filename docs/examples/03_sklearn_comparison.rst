Scikit-learn Algorithm Comparison
==================================

This example compares FuzzyART with popular scikit-learn classifiers on multiple datasets.

Overview
--------

We'll benchmark FuzzyART against:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
- Gradient Boosting

Across three datasets:

- Iris
- Wine
- Breast Cancer

Complete Comparison Code
------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, f1_score
    
    # Scikit-learn models
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # FuzzyART model
    from fuzzyart.models import FuzzyARTMAP
    
    # Load datasets
    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine(),
        'Breast Cancer': load_breast_cancer(),
    }
    
    # Define models
    models = {
        'FuzzyARTMAP': FuzzyARTMAP(rho=0.6, alpha=0.1, beta=0.7),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }
    
    # Evaluate on all datasets
    results = pd.DataFrame()
    
    for dataset_name, dataset in datasets.items():
        print(f"\nEvaluating on {dataset_name} dataset...")
        X = dataset.data
        y = dataset.target
        
        # Normalize
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        for model_name, model in models.items():
            # Cross-validation
            scores = cross_val_score(
                model, X_scaled, y, cv=5, scoring='accuracy'
            )
            
            results = pd.concat([results, pd.DataFrame({
                'Dataset': [dataset_name],
                'Model': [model_name],
                'Mean Accuracy': [scores.mean()],
                'Std Dev': [scores.std()],
            })], ignore_index=True)
            
            print(f"  {model_name}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Display results
    print("\n" + "="*60)
    print("Summary Results")
    print("="*60)
    print(results.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, dataset_name in enumerate(datasets.keys()):
        ax = axes[idx]
        data = results[results['Dataset'] == dataset_name]
        
        x_pos = np.arange(len(data))
        ax.bar(x_pos, data['Mean Accuracy'], 
               yerr=data['Std Dev'], capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data['Model'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset_name} Dataset')
        ax.set_ylim([0.5, 1.05])
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Algorithm Comparison: 5-Fold Cross-Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

Single Dataset Detailed Comparison
-----------------------------------

Detailed evaluation on one dataset:

.. code-block:: python

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    import seaborn as sns
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'FuzzyARTMAP': FuzzyARTMAP(rho=0.6),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf'),
        'Random Forest': RandomForestClassifier(n_estimators=100),
    }
    
    # Detailed metrics
    metrics = {}
    
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics[model_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        }
    
    # Display
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df.round(3))
    
    # Visualize
    metrics_df.plot(kind='bar', rot=45)
    plt.ylabel('Score')
    plt.title('Detailed Performance Metrics')
    plt.ylim([0.5, 1.05])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

Computational Efficiency Comparison
------------------------------------

Compare training time and prediction speed:

.. code-block:: python

    import time
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'FuzzyARTMAP': FuzzyARTMAP(rho=0.6),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(kernel='rbf'),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Logistic Regression': LogisticRegression(max_iter=1000),
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Training time
        start = time.time()
        for _ in range(100):
            model.fit(X_scaled, y)
        train_time = (time.time() - start) / 100
        
        # Prediction time
        start = time.time()
        for _ in range(1000):
            model.predict(X_scaled)
        pred_time = (time.time() - start) / 1000
        
        results[model_name] = {
            'Train Time (ms)': train_time * 1000,
            'Predict Time (μs)': pred_time * 1e6,
        }
    
    efficiency_df = pd.DataFrame(results).T
    print(efficiency_df.round(3))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(efficiency_df.index, efficiency_df['Train Time (ms)'])
    axes[0].set_ylabel('Training Time (ms)')
    axes[0].set_title('Training Speed')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(efficiency_df.index, efficiency_df['Predict Time (μs)'])
    axes[1].set_ylabel('Prediction Time (μs)')
    axes[1].set_title('Prediction Speed')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

Hyperparameter Sensitivity Analysis
------------------------------------

Compare sensitivity to hyperparameter changes:

.. code-block:: python

    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test FuzzyARTMAP with different rho values
    rho_values = np.arange(0.1, 1.0, 0.1)
    rho_scores = []
    
    for rho in rho_values:
        model = FuzzyARTMAP(rho=rho)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        rho_scores.append(scores.mean())
    
    # Compare with KNN's k parameter
    k_values = range(1, 11)
    k_scores = []
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(rho_values, rho_scores, marker='o')
    axes[0].set_xlabel('Vigilance Parameter (rho)')
    axes[0].set_ylabel('Cross-validation Accuracy')
    axes[0].set_title('FuzzyARTMAP Sensitivity to rho')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_values, k_scores, marker='s')
    axes[1].set_xlabel('Number of Neighbors (k)')
    axes[1].set_ylabel('Cross-validation Accuracy')
    axes[1].set_title('KNN Sensitivity to k')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

Advantages and Disadvantages Summary
-------------------------------------

.. code-block:: python

    advantages_disadvantages = {
        'FuzzyARTMAP': {
            'Advantages': [
                'Online learning capable',
                'No predefined cluster number',
                'Interpretable prototypes',
                'Fast prediction',
            ],
            'Disadvantages': [
                'Sensitive to hyperparameters',
                'Order-dependent clustering',
                'Limited Python ecosystem',
            ],
        },
        'KNN': {
            'Advantages': [
                'Simple and intuitive',
                'No training required',
                'Effective with small data',
            ],
            'Disadvantages': [
                'Slow prediction',
                'High memory usage',
                'Sensitive to feature scaling',
            ],
        },
        'SVM': {
            'Advantages': [
                'Effective in high dimensions',
                'Memory efficient',
                'Versatile kernel options',
            ],
            'Disadvantages': [
                'Requires careful scaling',
                'Difficult to interpret',
                'Slow on large datasets',
            ],
        },
        'Random Forest': {
            'Advantages': [
                'Handles mixed data types',
                'Feature importance available',
                'Robust to outliers',
            ],
            'Disadvantages': [
                'High memory usage',
                'Difficult to interpret',
                'Slow to train on large data',
            ],
        },
    }
    
    # Display comparison
    for model, info in advantages_disadvantages.items():
        print(f"\n{model}:")
        print("  Advantages:")
        for adv in info['Advantages']:
            print(f"    ✓ {adv}")
        print("  Disadvantages:")
        for dis in info['Disadvantages']:
            print(f"    ✗ {dis}")

Recommendation Guidelines
-------------------------

**Use FuzzyART when:**

- You need online learning capability
- Cluster interpretation is important
- You don't know the optimal number of clusters
- You have memory constraints
- Prediction speed is critical

**Use KNN when:**

- Dataset is small and feature space is low-dimensional
- You want a simple, interpretable baseline
- Similarity matching is more important than statistical modeling

**Use SVM when:**

- Data is high-dimensional
- You need a robust, well-tuned classifier
- Interpretability is less important

**Use Random Forest when:**

- Features are heterogeneous (mixed types)
- You need feature importance
- Training time is not critical
- You need strong generalization

**Use Logistic Regression when:**

- You need probabilistic predictions
- Interpretability is essential
- Linear relationships dominate

**Use Gradient Boosting when:**

- Maximum accuracy is needed
- You have sufficient training data
- Training time and complexity are acceptable

Key Takeaways
-------------

- FuzzyART is competitive with traditional methods
- Different algorithms excel in different scenarios
- Hyperparameter tuning is crucial for all methods
- No single algorithm is universally best
- Context and requirements should guide algorithm selection

Next Steps
----------

- See :doc:`01_iris_classification` for basic example
- Review :doc:`02_olivetti_faces` for advanced example
- Check :doc:`../guide/hyperparameters` for tuning strategies
- Explore :doc:`../api/models` for other available models
