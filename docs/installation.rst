Installation
=============

Prerequisites
-------------

- Python 3.8 or higher
- pip or poetry package manager

Installing from PyPI
--------------------

The easiest way to install FuzzyART is using pip:

.. code-block:: bash

    pip install fuzzyart

Installing with Poetry
----------------------

If you use Poetry for dependency management:

.. code-block:: bash

    poetry add fuzzyart

Or add it to your ``pyproject.toml``:

.. code-block:: toml

    [tool.poetry.dependencies]
    fuzzyart = "^1.0.0"

Then install with:

.. code-block:: bash

    poetry install

Installing from Source
----------------------

To install the development version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/yourusername/fuzzyart.git

Or clone the repository and install locally:

.. code-block:: bash

    git clone https://github.com/yourusername/fuzzyart.git
    cd fuzzyart
    pip install -e .

With Poetry:

.. code-block:: bash

    git clone https://github.com/yourusername/fuzzyart.git
    cd fuzzyart
    poetry install

Dependencies
------------

FuzzyART requires:

- NumPy: Numerical computing
- Scikit-learn: Machine learning utilities
- SciPy: Scientific computing

Optional dependencies for examples and visualization:

- Matplotlib: Data visualization
- Pandas: Data manipulation

Installing with Optional Dependencies
--------------------------------------

To install FuzzyART with all optional dependencies:

.. code-block:: bash

    pip install fuzzyart[dev]

Or with Poetry:

.. code-block:: bash

    poetry add fuzzyart[dev]

Verifying Installation
----------------------

To verify that FuzzyART is installed correctly, open a Python shell and run:

.. code-block:: python

    import fuzzyart
    print(fuzzyart.__version__)

If this runs without errors, FuzzyART is ready to use!

Troubleshooting
---------------

If you encounter issues during installation, please ensure:

1. You have Python 3.8+ installed
2. pip is up to date: ``pip install --upgrade pip``
3. You have administrator or user permissions to install packages
4. Your internet connection is stable

For Poetry users:

1. Poetry is properly installed: ``poetry --version``
2. Your virtual environment is activated: ``poetry shell``
3. Dependencies are up to date: ``poetry update``

For additional help, open an issue on the GitHub repository.
