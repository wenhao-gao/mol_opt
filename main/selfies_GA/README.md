# SELFIES-GA+D: Augmenting genetic algorithms with deep neural networks for exploring the chemical space
This repository contains code for the paper: [Augmenting genetic algorithms with deep neural networks for exploring the chemical space](https://arxiv.org/abs/1909.11655). 



## Prerequisites

Before running the code, please ensure you have the following:
- [SELFIES (any version)](https://github.com/aspuru-guzik-group/selfies): `pip install selfies`
  The code was run with v0.1.1 (which is the fastest), however, the code is compatible with any version. 
- [RDKit](https://www.rdkit.org/docs/Install.html)
- [tensorboardX](https://pypi.org/project/tensorboardX/)
- [Pytorch v0.4.1](https://pytorch.org/)
- [Python 3.0 or up](https://www.python.org/download/releases/3.0/)
- [numpy](https://pypi.org/project/numpy/)



## run the code

The following settings can be customized (found at the end of the file 'core_GA.py'): 
- num_generations: Number of generations to run the GA
- generation_size: Molecular population size encountered in each generation 
- starting_selfies: Initial population of molecules 
- max_molecules_len: Length of the largest molecule string
- disc_epochs_per_generation: Number of epochs of training the discriminator neural network 
- disc_enc_type: Type of molecular encoding shown to the discriminator
- disc_layers : Discriminator architecture
- training_start_gen: generation after which discriminator training begins 
- device: Device the discriminator is trained on 
- properties_calc_ls: Property evaluations to be completed for each molecule of the GA
- num_processors: Number of cpu cores to parallelize calculations over
- beta: Value of parameter beta
- impose_time_adapted_pen: Boolean variable to indicated use of a time-adapted discriminator penalty



















