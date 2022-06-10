# mol_opt: A Benchmark for Practical Molecular Optimization

This repository created an open-source benchmark for Practical Molecular Optimization, (PMO), to facilitate the transparent and reproducible evaluation of algorithmic advances in molecular optimization. This repository supports 25 molecular design algorithms on 23 tasks with a particular focus on sample efficiency (oracle calls). 



## install 

```bash
conda create -n molopt python=3.7
conda activate molopt 
pip install torch 
pip install PyTDC 
pip install PyYAML
conda install -c rdkit rdkit 
pip install wandb   
wandb login  ### user need to register wandb
```

We recommend to use PyTorch 1.10.2 and PyTDC 0.3.6. 

<!-- pip install nltk: only for smiles_ga  -->
<!-- pip install guacamol  -->
<!-- pip install networkx  -->
<!-- pip install joblib  -->



## activate conda environment 

```bash
conda activate molopt 
```



## 25 Models

|                    | `runable` | `compatible` | `additional package` | `test` | `clean` |
|--------------------|-----------|--------------|----------|--------|---------|
| **screening**      | ✅        | ✅           | -        |        |         |
| **molpal**         | ✅        | ✅           | ray      |        |         |
| **graph\_ga**      | ✅        | ✅           | joblib   |        |         |
| **smiles\_ga**     | ✅        | ✅           | joblib, nltk   |        |         |
| **stoned**         | ✅        | ✅           | -         |        |         |
| **selfies\_ga**    | ✅        | ✅           | selfies   |        |         |
| **graph\_mcts**    | ✅        | ✅           | -       |        |         |
| **smiles\_lstm\_hc**   | ✅    | ✅           | guacamol         |        |         |
| **selfies\_lstm\_hc**  | ✅    | ✅           | guacamol, selfies         |        |         |
| **smiles\_vae**    | ✅        | ✅           | botorch         |        |         |
| **selfies\_vae**   | ✅        | ✅           | botorch, selfies         |        |         |
| **jt\_vae**        | ✅        | ✅           | botorch          |        |         |
| **gpbo**           | ✅        | ✅           | botorch, networkx         |        |         |
| **reinvent**       | ✅        | ✅           | -         |        |         |
| **reinvent\_selfies** | ✅     | ✅           | selfies         |        |         |
| **moldqn**         | ✅        | ✅           | networks, requests    |        |         |
| **mimosa**         | ✅        | ✅           | -         |        |         |
| **mars**           | ✅        | ✅           | chemprop, networkx, dgl         |        |         |
| **dog\_gen**       | ✅        | ✅           | **super complex**         |        |         |
| **dog\_ae**        | ✅        | ✅           | **super complex**         |        |         |
| **synnet**         | ✅        | ✅           | dgl, pytorch_lightning, networkx, matplotlib        |        |         |
| **pasithea**       | ✅        | ✅           | selfies, matplotlib         |        |         |
| **dst**            | ✅        | ✅           | -         |        |         |
| **gflownet**       | ✅        | ✅           | torch_geometric, torch_sparse, pdb         |        |         |
| **gflownet\_al**   | ✅        | ✅           | torch_geometric, torch_sparse, pdb         |        |         ||


## Run with one-line bash command line

There are three types of runs defined in our code base: 
* `simple`: A single run for testing purposes for each oracle, is the defualt.
* `production`: Multiple independent runs with various random seeds for each oracle.
* `tune`: A hyper-parameter tuning over the search space defined in `main/MODEL_NAME/hparam_tune.yaml` for each oracle.

```bash
## run a single test run on qed with wandb logging online
python run.py MODEL_NAME --wandb online
## specify multiple random seeds 
python run.py MODEL_NAME --seed 0 1 2 
## run 5 runs with different random seeds on multuple oracles with wandb logging offline
python run.py MODEL_NAME --task production --n_runs 5 --oracles qed 
## run a hyper-parameter tuning starting from smiles in a smi_file, 30 runs in total without wandb logging
python run.py MODEL_NAME --task tune --n_runs 30 --smi_file XX --wandb disabled --other_args XX 
```


## Hyperparameters

We separate hyperparameters for task-level control, defined from `argparse`, and algorithm-level control, defined from `hparam_default.yaml`. There is no clear boundary for them, but we recommend one keep all hyperparameters in the `self._optimize` function as task-level. 

* **running hyperparameter**: parser argument. 
* **default model hyperparameter**: `hparam_default.yaml`
* **tuning model hyperparameter**: `hparam_tune.yaml` 

For algorithm-level hyperparameters, we adopt the stratforward yaml file format. One should define a default set of hyper-parameters in `main/MODEL_NAME/hparam_default.yaml`:

```python
population_size: 50
offspring_size: 100
mutation_rate: 0.02
patience: 5
max_generations: 1000
```

And the search space for hyper-parameter tuning in `main/MODEL_NAME/hparam_tune.yaml`:

```python
name: graph_ga
method: random
metric:
  goal: maximize
  name: avg_top100
parameters:
  population_size:
    values: [20, 40, 50, 60, 80, 100, 150, 200]
  offspring_size:
    values: [50, 100, 200, 300]
  mutation_rate:
    distribution: uniform
    min: 0
    max: 0.1
  patience:
    value: 5
  max_generations:
    value: 1000
```

We use the [sweep](https://docs.wandb.ai/guides/sweeps) function in [wandb](https://docs.wandb.ai) for a convenient visualization. The yaml file should follow the format as above. Further detail is in this [instruction](https://docs.wandb.ai/guides/sweeps/configuration).



## How to Contribute to our benchmark

Our repository is an open-source initiative. To get involved, check our [Contribution Guidelines](CONTRIBUTE.md)!









