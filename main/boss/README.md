# BOSS Package
This is a package for performing Bayesian Optimization over String Spaces (BOSS). It accompanies https://arxiv.org/pdf/2010.00979.pdf and provides notebooks to recreate all the experiments.


The code is built upon the emukit Bayesian optimization library. We recommend following their tutorials to get started (https://github.com/emukit/emukit/tree/master/notebooks)


We currently support the following spaces:

1) unconstrained strings of fixed-length
2) locally-constrained strings of fixed-length
3) strings of varied length following constraints given by a context-free grammar
4) a candidate set of strings of varied length

and provide implementations for the following surrogate models:

1) Gaussian process with a linear kernel applied to a one-hot-encoding of strings
2) Gaussian process with an RBF kernel applied to a bag-of-ngrams representation of strings
3) Gaussian process with an SSK kernel
4) Gaussian process with a split SSK kernel (for scaling SSK to long strings)
5) Random search


We also provide GPU support for string kernel GPs through GPflow. For an example see 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/henrymoss/BOSS/blob/master/example_notebooks/Molecule_prediction_demo.ipynb)
