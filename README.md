# mol_opt: A Benchmark for Practical Molecular Optimization

---

[![GitHub Repo stars](https://img.shields.io/github/stars/wenhao-gao/mol_opt)](https://github.com/wenhao-gao/mol_opt/stargazers)
[![GitHub Repo forks](https://img.shields.io/github/forks/wenhao-gao/mol_opt)](https://github.com/wenhao-gao/mol_opt/network/members)


This repository hosts an open-source benchmark for Practical Molecular Optimization (**PMO**), to facilitate the transparent and reproducible evaluation of algorithmic advances in molecular optimization. This repository supports 25 molecular design algorithms on 23 tasks with a particular focus on sample efficiency (oracle calls). The preprint version of the paper is available at https://arxiv.org/pdf/2206.12411.pdf



## Installation 

```bash
conda create -n molopt python=3.7
conda activate molopt 
pip install torch 
pip install PyTDC 
pip install PyYAML
conda install -c rdkit rdkit 
```

We recommend to use PyTorch 1.10.2 and PyTDC 0.3.6. 

<!-- pip install guacamol  -->
<!-- pip install networkx  -->
<!-- pip install joblib  -->



Then we can activate conda via following command. 
```bash
conda activate molopt 
```



## 29 Methods


Based the ML methodologies, all the methods are categorized into: 
* virtual screening
    * **screening** randomly search ZINC database. 
    * **molpal** uses molecular property predictor to prioritize the high-scored molecules. 
* GA (genetic algorithm)
    * **graph\_ga** based on molecular graph.
    * **smiles\_ga** based on SMILES 
    * **selfies\_ga** based on SELFIES
    * **stoned** based on SELFIES
    * **synnet** based on synthesis
* VAE (variational auto-encoder)
    * **smiles\_vae** based on SMILES
    * **selfies\_vae** based on SELFIES
    * **jt\_vae** based on junction tree (fragment as building block)
    * **dog\_ae** based on synthesis 
* BO (Bayesian optimization)
    * **gpbo** 
* RL (reinforcement learning)
    * **reinvent** 
    * **reinvent\_selfies** 
    * **graphinvent** 
    * **moldqn** 
    * **smiles_aug_mem**
    * **smiles_bar**
* HC (hill climbing)
    * **smiles\_lstm\_hc** is SMILES-level HC. 
    * **smiles\_ahc** is SMILES-level augmented HC. 
    * **selfies\_lstm\_hc** is SELFIES-level HC
    * **mimosa** is graph-level HC
    * **dog\_gen** is synthesis based HC 
* gradient (gradient ascent)
    * **dst** is based molecular graph. 
    * **pasithea** is based on SELFIES. 
* SBM (score-based modeling)
    * **gflownet**  
    * **gflownet\_al** 
    * **mars** 

`time` is the average rough clock time for a single run in our benchmark and do not involve the time for pretraining and data preprocess. 
We have processed the data, pretrained the model. Both are available in the repository. 

|                                                                                                           | `assembly` | `additional package`                          | `time`    | `requires_gpu` |
|-----------------------------------------------------------------------------------------------------------|------------|-----------------------------------------------|-----------|---------|
| **screening**                                                                                             | -          | -                                             | 2 min     |    no     |
| **[molpal](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc06805e)**                              | -          | ray, tensorflow, ConfigArgParse, pytorch-lightning        | 1 hour    |    no     |
| **[graph\_ga](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)**                        | fragment   | joblib                                        | 3 min     |   no    |
| **[smiles\_ga](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839)**                                       | SMILES     | joblib, nltk                                  | 2 min     |    no     |
| **[stoned](https://chemrxiv.org/engage/chemrxiv/article-details/60c753f00f50db6830397c37)**               | SELFIES    | -                                             | 3 min     |    no    |
| **[selfies\_ga](https://openreview.net/forum?id=H1lmyRNFvr)**                                             | SELFIES    | selfies                                       | 20 min    |    no     |
| **[graph\_mcts](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)**                      | atom       | -                                             | 2 min     |    no     |
| **[smiles\_lstm\_hc](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839)**                                 | SMILES     | guacamol                                      | 4 min     |    no     |
| **[smiles\_ahc](https://arxiv.org/pdf/2212.01385.pdf)**                                                   | SMILES     |                                               | 4 min     |    no     |
| **[selfies\_lstm\_hc](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839)**                                | SELFIES    | guacamol, selfies                             | 4 min     |    yes    |
| **[smiles\_vae](https://arxiv.org/pdf/1610.02415.pdf)**                                                   | SMILES     | botorch                                       | 20 min    |    yes     |
| **[selfies\_vae](https://arxiv.org/pdf/1610.02415.pdf)**                                                  | SELFIES    | botorch, selfies                              | 20 min    |    yes     |
| **[jt\_vae](https://arxiv.org/pdf/1802.04364.pdf)**                                                       | fragment   | botorch                                       | 20 min    |    yes     |
| **[gpbo](https://openreview.net/forum?id=gS3XMun4cl_)**                                                   | fragment   | botorch, networkx                             | 15 min    |    no     |
| **[reinvent](https://arxiv.org/abs/1704.07555)**                                                          | SMILES     | pexpect, bokeh                                | 2 min     |    yes    |
| **[reinvent\_selfies](https://arxiv.org/abs/1704.07555)**                                                 | SELFIES    | selfies, pexpect, bokeh                       | 3 min     |    yes     |
| **[smiles\_aug\_mem](https://chemrxiv.org/engage/chemrxiv/article-details/6464dc3ea32ceeff2dcbd948)**     | SMILES     | reinvent-models==0.0.15rc1                    | 2 min     |    yes     |
| **[smiles\_bar](https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00838)**                                 | SMILES     | reinvent-models==0.0.15rc1                    | 2 min     |    yes     |
| **[reinvent\_selfies](https://arxiv.org/abs/1704.07555)**                                                 | SELFIES    | selfies                                       | 3 min     |    yes     |
| **[moldqn](https://www.nature.com/articles/s41598-019-47148-x?ref=https://githubhelp.com)**               | atom       | networks, requests                            | 60 min    |     yes    |
| **[mimosa](https://arxiv.org/abs/2010.02318)**                                                            | fragment   | -                                             | 10 min    |     yes    |
| **[mars](https://openreview.net/pdf?id=kHSu4ebxFXY)**                                                     | fragment   | chemprop, networkx, dgl                       | 20 min    |    yes     |
| **[dog\_gen](https://proceedings.neurips.cc/paper/2020/file/4cc05b35c2f937c5bd9e7d41d3686fff-Paper.pdf)** | synthesis  | extra conda                                   | 120 min   |     yes    |
| **[dog\_ae](https://proceedings.neurips.cc/paper/2020/file/4cc05b35c2f937c5bd9e7d41d3686fff-Paper.pdf)**  | synthesis  | extra conda                                   | 50 min    |    yes     |
| **[synnet](https://openreview.net/forum?id=FRxhHdnxt1)**                                                  | synthesis  | dgl, pytorch_lightning, networkx, matplotlib  | 2-5 hours |    yes     |
| **[pasithea](https://arxiv.org/pdf/2012.09712.pdf)**                                                      | SELFIES    | selfies, matplotlib                           | 50 min    |    yes     |
| **[dst](https://openreview.net/pdf?id=w_drCosT76)**                                                       | fragment   | -                                             | 120 min   |    no     |
| **[gflownet](https://arxiv.org/abs/2106.04399)**                                                          | fragment   | torch_{geometric,sparse,cluster}, pdb         | 30 min    |     yes    |
| **[gflownet\_al](https://arxiv.org/abs/2106.04399)**                                                      | fragment   | torch_{geometric,sparse,cluster}, pdb         | 30 min    |    yes     ||


## Run with one-line code

There are three types of runs defined in our code base: 
* `simple`: A single run for testing purposes for each oracle, is the defualt.
* `production`: Multiple independent runs with various random seeds for each oracle.
* `tune`: A hyper-parameter tuning over the search space defined in `main/MODEL_NAME/hparam_tune.yaml` for each oracle.

```bash
## specify multiple random seeds 
python run.py MODEL_NAME --seed 0 1 2 
## run 5 runs with different random seeds with specific oracle 
python run.py MODEL_NAME --task production --n_runs 5 --oracles qed 
## run a hyper-parameter tuning starting from smiles in a smi_file, 30 runs in total
python run.py MODEL_NAME --task tune --n_runs 30 --smi_file XX --other_args XX 
```

`MODEL_NAME` are listed in the table above. 

## Multi-Objective Optimization

Multi-objective optimization is implemented in `multiobjective` branch. We use "+" to connect multiple properties, please see the command line below. 

```bash
python run.py MODEL_NAME --oracles qed+jnk3  
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




## Contribute

Our repository is an open-source initiative. To update a better set of parameters or incldue your model in out benchmark, check our [Contribution Guidelines](CONTRIBUTE.md)!



