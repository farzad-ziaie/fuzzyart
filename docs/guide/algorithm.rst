Algorithm Guide
===============

Overview
--------

Fuzzy Adaptive Resonance Theory (Fuzzy ART) is a neural network architecture that performs unsupervised learning through a dynamic clustering process. This guide explains the core concepts and algorithms implemented in FuzzyART.

Adaptive Resonance Theory (ART)
-------------------------------

ART networks solve the stability-plasticity dilemma:

- **Stability**: The network retains previously learned information
- **Plasticity**: The network can learn new patterns

ART achieves this balance through a resonance mechanism that compares incoming patterns with learned prototypes.

Fuzzy ART Components
--------------------

Fuzzy ART networks consist of two layers:

1. **F1 Layer (Input Field)**: Receives and processes input patterns
2. **F2 Layer (Cluster Field)**: Stores learned cluster prototypes

The network implements:

- **Complement Coding**: Encodes inputs with their complements to enable symmetric similarity measures
- **Choice Function**: Selects the best matching cluster based on similarity
- **Vigilance Test**: Determines whether the input sufficiently matches the chosen cluster
- **Learning**: Updates cluster weights when resonance is achieved

Key Concepts
~~~~~~~~~~~~

Vigilance Parameter (ρ)
^^^^^^^^^^^^^^^^^^^^^^^

The vigilance parameter controls clustering resolution:

- Low ρ (e.g., 0.1): Coarse clusters, fewer categories
- High ρ (e.g., 0.9): Fine-grained clusters, many categories

Choice Function
^^^^^^^^^^^^^^^

The choice function selects the best matching cluster:

.. math::

    T_j = \frac{|I \land w_j|}{\\alpha + |w_j|}

Where:

- I is the input pattern
- w_j is the j-th cluster weight vector
- ∧ is the fuzzy AND operation (minimum)
- α is the choice parameter

Vigilance Test
^^^^^^^^^^^^^^

After selecting a cluster, the network tests whether the match is sufficiently good:

.. math::

    \frac{|I \land w_j|}{|I|} \geq \rho

If this condition is satisfied, resonance occurs and learning happens. Otherwise, the network searches for another cluster or creates a new one.

Fuzzy ARTMAP
------------

Fuzzy ARTMAP extends Fuzzy ART for supervised learning:

- Uses two Fuzzy ART modules (one for inputs, one for outputs)
- Implements match tracking to improve generalization
- Automatically determines vigilance through prediction errors

Semi-Supervised Learning
-------------------------

Semi-supervised ARTMAP extends Fuzzy ARTMAP to leverage both labeled and unlabeled data:

- Labeled data guides cluster formation
- Unlabeled data refines cluster boundaries
- Useful when labeled data is scarce

Bayesian ARTMAP
---------------

Bayesian ARTMAP incorporates probabilistic reasoning:

- Models cluster uncertainty
- Provides confidence estimates
- Maintains category statistics for principled decisions

Learning Process
----------------

1. **Initialization**: Create empty cluster field
2. **Input**: Present pattern to network
3. **Choice**: Find best matching cluster using choice function
4. **Vigilance**: Test if match meets vigilance criterion
5. **Resonance or Search**: If match satisfies vigilance, learn (resonance); otherwise search for another cluster
6. **Learning**: Update cluster weights using:

   .. math::

       w_j^{new} = \beta (I \land w_j^{old}) + (1-\beta) w_j^{old}

   Where β is the learning rate

7. **Repeat**: Continue with next pattern

Computational Complexity
------------------------

- **Training**: O(n·m·k) where n = samples, m = features, k = clusters
- **Prediction**: O(m·k) per sample
- **Memory**: O(m·k) for storing cluster prototypes

Advantages and Limitations
---------------------------

Advantages
~~~~~~~~~~

- Online learning capability
- No predefined number of clusters
- Interpretable cluster prototypes
- Relatively fast computation

Limitations
~~~~~~~~~~~

- Sensitive to hyperparameter values (especially ρ)
- Cluster creation depends on input order
- Limited theoretical understanding of some variants

Further Reading
---------------

For more detailed information, see the references and academic papers cited in the main documentation.
