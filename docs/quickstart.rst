Quick Start
===========

Basic Usage
-----------

Here's a simple example to get started with FuzzyART:

.. code-block:: python

    from fuzzyart.models import FuzzyART
    import numpy as np
    
    # Generate sample data
    X = np.random.rand(100, 2)
    
    # Create and fit model
    model = FuzzyART(rho=0.5)
    model.fit(X)
    
    # Make predictions
    labels = model.predict(X)
    print(f"Number of clusters: {len(np.unique(labels))}")

Classification Example
----------------------

For supervised learning with Fuzzy ARTMAP:

.. code-block:: python

    from fuzzyart.models import FuzzyARTMAP
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Load data
    data = load_iris()
    X = data.data
    y = data.target
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train model
    model = FuzzyARTMAP(rho=0.6, alpha=0.1, beta=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2%}")

Common Parameters
-----------------

Most FuzzyART models share these key hyperparameters:

- **rho**: Vigilance parameter (0-1). Controls cluster granularity. Higher values create finer clusters.
- **alpha**: Choice parameter. Controls learning rate during resonance.
- **beta**: Learning rate (0-1). Controls how much weights are updated.

More details on hyperparameters can be found in the :doc:`guide/hyperparameters` guide.

Next Steps
----------

- Explore :doc:`guide/algorithm` for theoretical background
- Check the :doc:`guide/preprocessing` guide for data preparation
- Review :doc:`examples/01_iris_classification` for a complete example
- See :doc:`api/models` for the full API reference
