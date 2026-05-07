Contributing Guidelines
=======================

Thank you for your interest in contributing to FuzzyART! This document provides guidelines and instructions for contributing.

Ways to Contribute
------------------

There are many ways to contribute to FuzzyART:

- **Code**: Submit bug fixes, new features, or improvements
- **Documentation**: Improve documentation, write examples, fix typos
- **Testing**: Write tests, report bugs, verify fixes
- **Discussion**: Answer questions, provide feedback, suggest improvements

Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment for all contributors.

- Be respectful and professional
- Avoid discriminatory language
- Welcome diverse perspectives
- Focus on constructive feedback

Any violations should be reported to the project maintainers.

Getting Started
---------------

1. Fork the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/yourusername/fuzzyart.git
    cd fuzzyart

2. Create a Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e ".[dev]"

3. Create a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git checkout -b feature/your-feature-name

4. Make Your Changes
~~~~~~~~~~~~~~~~~~~~

Implement your changes while following the coding standards (see below).

5. Write Tests
~~~~~~~~~~~~~~

Add tests for new functionality:

.. code-block:: bash

    pytest tests/

6. Update Documentation
~~~~~~~~~~~~~~~~~~~~~~~

Update relevant documentation files and docstrings.

7. Commit and Push
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git add .
    git commit -m "Brief description of changes"
    git push origin feature/your-feature-name

8. Submit a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~

Open a pull request on GitHub with a clear description of your changes.

Coding Standards
----------------

Style Guide
~~~~~~~~~~~

We follow PEP 8 conventions with some customizations:

- Line length: 88 characters (Black formatter)
- Use type hints for function signatures
- Document all public functions and classes

**Example:**

.. code-block:: python

    def fuzzy_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute fuzzy AND operation (element-wise minimum).
        
        Parameters
        ----------
        a : np.ndarray
            First operand
        b : np.ndarray
            Second operand
            
        Returns
        -------
        np.ndarray
            Element-wise minimum of a and b
        """
        return np.minimum(a, b)

Code Formatting
~~~~~~~~~~~~~~~

Use Black for automatic formatting:

.. code-block:: bash

    black fuzzyart/

Use isort for import organization:

.. code-block:: bash

    isort fuzzyart/

Use flake8 for linting:

.. code-block:: bash

    flake8 fuzzyart/

Docstring Format
~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

    class FuzzyART:
        """
        Fuzzy Adaptive Resonance Theory clustering.
        
        Parameters
        ----------
        rho : float
            Vigilance parameter (0-1)
        alpha : float
            Choice parameter (default: 0.0)
        beta : float
            Learning rate (default: 1.0)
            
        Attributes
        ----------
        weights_ : np.ndarray
            Learned cluster prototypes
        labels_ : np.ndarray
            Cluster assignments
            
        Examples
        --------
        >>> from fuzzyart.models import FuzzyART
        >>> import numpy as np
        >>> X = np.random.rand(100, 2)
        >>> model = FuzzyART(rho=0.5)
        >>> labels = model.fit_predict(X)
        """

Testing Requirements
--------------------

Unit Tests
~~~~~~~~~~

All code changes should include corresponding unit tests:

.. code-block:: python

    import pytest
    import numpy as np
    from fuzzyart.models import FuzzyART
    
    class TestFuzzyART:
        """Tests for FuzzyART class."""
        
        def test_initialization(self):
            """Test model initialization."""
            model = FuzzyART(rho=0.5)
            assert model.rho == 0.5
        
        def test_fit_predict(self):
            """Test fit and predict methods."""
            X = np.random.rand(50, 3)
            model = FuzzyART(rho=0.5)
            labels = model.fit_predict(X)
            assert len(labels) == len(X)
            assert len(np.unique(labels)) > 0
        
        def test_invalid_rho(self):
            """Test validation of rho parameter."""
            with pytest.raises(ValueError):
                FuzzyART(rho=1.5)  # rho must be in [0, 1]

Run Tests
~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest tests/
    
    # Run with coverage
    pytest --cov=fuzzyart tests/
    
    # Run specific test
    pytest tests/test_models.py::TestFuzzyART::test_fit_predict

Test Coverage
~~~~~~~~~~~~~

Aim for >85% code coverage:

.. code-block:: bash

    pytest --cov=fuzzyart --cov-report=html tests/

Documentation Requirements
---------------------------

Docstring Changes
~~~~~~~~~~~~~~~~~

When modifying function/class docstrings:

1. Keep descriptions clear and concise
2. Include all parameters
3. Document return values
4. Add examples where helpful
5. Use proper reStructuredText formatting

Example Documentation
~~~~~~~~~~~~~~~~~~~~~

Adding new examples:

1. Create a new file in ``examples/`` (e.g., ``04_custom_application.py``)
2. Write well-commented, runnable code
3. Include output and visualizations
4. Add corresponding ``.rst`` file for documentation
5. Update ``docs/index.rst`` to include the new example

Comment Guidelines
~~~~~~~~~~~~~~~~~~

Use comments for:

- Complex algorithms
- Non-obvious design decisions
- Workarounds or hacks
- Algorithm references

Example:

.. code-block:: python

    # Complement coding: concatenate x with (1-x)
    # Reference: Carpenter & Grossberg (1987)
    X_complement = np.hstack([X, 1 - X])

Submitting Pull Requests
------------------------

PR Title Format
~~~~~~~~~~~~~~~

Use descriptive titles:

- Good: "Add online learning support to FuzzyARTMAP"
- Bad: "Update code"

PR Description Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ## Description
    Brief description of the changes.
    
    ## Type of Change
    - [ ] Bug fix
    - [ ] New feature
    - [ ] Documentation update
    - [ ] Performance improvement
    
    ## Related Issues
    Fixes #123
    
    ## Testing
    Describe test coverage.
    
    ## Checklist
    - [ ] Code follows style guidelines
    - [ ] Tests added/updated
    - [ ] Documentation updated
    - [ ] No new warnings

Review Process
~~~~~~~~~~~~~~

1. Maintainers will review your PR
2. Respond to feedback constructively
3. Update based on comments
4. Once approved, your PR will be merged

Reporting Bugs
--------------

Creating a Good Bug Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When reporting a bug, include:

1. **Title**: Clear, descriptive title
2. **Environment**: Python version, OS, library versions
3. **Steps to Reproduce**: Minimal code example
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happened
6. **Traceback**: Full error message if applicable
7. **Additional Context**: Any other relevant information

Example Bug Report
~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    Title: FuzzyART fails with NaN weights on certain inputs
    
    Environment:
    - Python 3.9
    - fuzzyart 1.0.0
    - numpy 1.21.0
    - scikit-learn 0.24.0
    
    Steps to Reproduce:
    ```python
    import numpy as np
    from fuzzyart.models import FuzzyART
    
    X = np.zeros((10, 2))  # All zeros
    model = FuzzyART(rho=0.5)
    model.fit(X)  # NaN error occurs here
    ```
    
    Expected: Model should handle zero inputs gracefully
    Actual: ValueError: invalid value encountered in divide
    
    Traceback: [full traceback]

Suggesting Enhancements
-----------------------

When suggesting a feature:

1. **Title**: Clear, specific title
2. **Motivation**: Why is this needed?
3. **Proposed Solution**: How should it work?
4. **Alternatives Considered**: Other approaches
5. **Additional Context**: References, examples

Development Workflow
--------------------

Local Development
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create and activate environment
    python -m venv venv
    source venv/bin/activate
    
    # Install in development mode
    pip install -e ".[dev]"
    
    # Make changes
    # ... edit files ...
    
    # Format code
    black fuzzyart/
    isort fuzzyart/
    
    # Run linter
    flake8 fuzzyart/
    
    # Run tests
    pytest tests/
    
    # Check coverage
    pytest --cov=fuzzyart tests/

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd docs
    make clean html
    # Open _build/html/index.html in browser

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Set up automatic checks before commits:

.. code-block:: bash

    pip install pre-commit
    pre-commit install

Releasing New Versions
----------------------

Version Numbering
~~~~~~~~~~~~~~~~~

We use Semantic Versioning: MAJOR.MINOR.PATCH

- MAJOR: Incompatible API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

Release Checklist
~~~~~~~~~~~~~~~~~

(For maintainers)

1. Update version in ``__init__.py``
2. Update ``CHANGELOG.rst``
3. Run full test suite
4. Build documentation
5. Create git tag
6. Build and upload to PyPI
7. Create GitHub release

Getting Help
------------

- **Questions**: Use GitHub Discussions
- **Issues**: Use GitHub Issues for bugs
- **Chat**: Join our community chat (if available)
- **Email**: Contact maintainers directly

Recognition
-----------

All contributors are recognized in:

- This file (CONTRIBUTING.rst)
- Release notes
- GitHub contributions page

Thank You
---------

Thank you for contributing to FuzzyART! Your efforts help make this project better for everyone.

Additional Resources
--------------------

- `GitHub Contributing Guide <https://docs.github.com/en/github/building-a-strong-community/setting-up-your-project-for-healthy-contributions>`_
- `NumPy Contribution Guide <https://numpy.org/doc/stable/reference/development/contributing.html>`_
- `Scikit-learn Contributing <https://scikit-learn.org/stable/developers/contributing.html>`_
- `PEP 8 Style Guide <https://pep8.org/>`_
