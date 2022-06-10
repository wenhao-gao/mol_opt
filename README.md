# mol_opt: A Benchmark for Practical Molecular Optimization

This repository hosts an open-source benchmark for Practical Molecular Optimization (**PMO**), to facilitate the transparent and reproducible evaluation of algorithmic advances in molecular optimization. This repository supports 25 molecular design algorithms on 23 tasks with a particular focus on sample efficiency (oracle calls). 



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



## activate conda 

```bash
conda activate molopt 
```



## 25 Models

`time` is the average rough clock time for a single run in our benchmark and do not involve the time for pretraining and data preprocess. 
We have processed the data, pretrained the model. Both are available in the repository. 

|                    | `runable` | `additional package` | `time` | `clean` |
|--------------------|-----------|----------|--------|---------|
| **screening**      | ✅        | -        |  2 min     |         |
| **molpal**         | ✅        | ray      |     ?   |         |
| **graph\_ga**      | ✅        | joblib   |  3 min      |         |
| **smiles\_ga**     | ✅        | joblib, nltk   |   2 min     |         |
| **stoned**         | ✅        | -         |   3 min     |         |
| **selfies\_ga**    | ✅        | selfies   |  20 min      |         |
| **graph\_mcts**    | ✅        | -       |   2 min     |         |
| **smiles\_lstm\_hc**   | ✅    | guacamol         |    4 min    |         |
| **selfies\_lstm\_hc**  | ✅    | guacamol, selfies         |    4 min    |         |
| **smiles\_vae**    | ✅        | botorch         |   20 min    |         |
| **selfies\_vae**   | ✅        | botorch, selfies         |    20 min    |         |
| **jt\_vae**        | ✅        | botorch          |    20 min    |         |
| **gpbo**           | ✅        | botorch, networkx         |    15 min    |         |
| **reinvent**       | ✅        | -         |    2 min    |         |
| **reinvent\_selfies** | ✅     | selfies         |    3 min    |         |
| **moldqn**         | ✅        | networks, requests    |    60 min    |         |
| **mimosa**         | ✅        | -         |    10 min    |         |
| **mars**           | ✅        | chemprop, networkx, dgl         |    20 min    |         |
| **dog\_gen**       | ✅        | extra conda        |    120 min    |         |
| **dog\_ae**        | ✅        | extra conda        |        |         |
| **synnet**         | ✅        | dgl, pytorch_lightning, networkx, matplotlib        |        |         |
| **pasithea**       | ✅        | selfies, matplotlib         |    50 min    |         |
| **dst**            | ✅        | -         |    120 min     |         |
| **gflownet**       | ✅        | torch_geometric, torch_sparse, torch_cluster, pdb        |    30 min    |         |
| **gflownet\_al**   | ✅        | torch_geometric, torch_sparse, torch_cluster,pdb         |    30 min    |         ||


## Run with one-line code

There are three types of runs defined in our code base: 
* `simple`: A single run for testing purposes for each oracle, is the defualt.
* `production`: Multiple independent runs with various random seeds for each oracle.
* `tune`: A hyper-parameter tuning over the search space defined in `main/MODEL_NAME/hparam_tune.yaml` for each oracle.

```bash
## run a single test run on qed with wandb logging online
python run.py MODEL_NAME --wandb online
## specify multiple random seeds 
python run.py MODEL_NAME --seed 0 1 2 
## run 5 runs with different random seeds with specific oracle with wandb logging offline
python run.py MODEL_NAME --task production --n_runs 5 --oracles qed 
## run a hyper-parameter tuning starting from smiles in a smi_file, 30 runs in total without wandb logging
python run.py MODEL_NAME --task tune --n_runs 30 --smi_file XX --wandb disabled --other_args XX 
```

`MODEL_NAME` are listed in the table above. 


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



## Contribute

Our repository is an open-source initiative. To get involved, check our [Contribution Guidelines](CONTRIBUTE.md)!









