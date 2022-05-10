# DST: Differentiable Scaffolding Tree for Molecule Optimization 



## 1. Installation 

To install locally, we recommend to install from `pip` and `conda`. Please see `conda.yml` for the package dependency. 
```bash
conda create -n dst python=3.7 
conda activate dst
pip install torch 
pip install PyTDC 
conda install -c rdkit rdkit 
```


Activate conda environment. 
```bash
conda activate dst
```

make directory
```bash
mkdir -p pretrained_model result 
```


## 2. Data and Setup
In our setup, we restrict the number of oracle calls. In realistic discovery settings, the oracle acquisition cost is usually not negligible. 


### Raw Data 
We use [`ZINC`](https://tdcommons.ai/generation_tasks/molgen/) database, which contains around 250K drug-like molecules and can be downloaded [`download ZINC`](https://tdcommons.ai/generation_tasks/molgen/). 
```bash
python download.py
```
- output
  - `data/zinc.tab`: all the smiles in ZINC, around 250K. 

### Oracle
Oracle is a property evaluator and is a function whose input is molecular structure, and output is the property. 
We consider following oracles: 
* `JNK3`: biological activity to JNK3, ranging from 0 to 1.
* `GSK3B` biological activity to GSK3B, ranging from 0 to 1. 
* `QED`: Quantitative Estimate of Drug-likeness, ranging from 0 to 1. 
* `SA`: Synthetic Accessibility, we normalize SA to (0,1). 
* `LogP`: solubility and synthetic accessibility of a compound. It ranges from negative infinity to positive infinity. 

For all the property scores above, higher is more desirable. 

### Optimization Task 
There are two kinds of optimization tasks: single-objective and multi-objective optimization. 
Multi-objective optimization contains `jnkgsk` (JNK3 + GSK3B), `qedsajnkgsk` (QED + SA + JNK3 + GSK3B). 


### Generate Vocabulary 
In this project, the basic unit is `substructure`, which can be atoms or single rings. 
The vocabulary is the set of frequent `substructures`. 
```bash 
python vocabulary.py
```
- input
  - `data/zinc.tab`: all the smiles in ZINC, around 250K. 
- output
  - `data/substructure.txt`: including all the substructures in ZINC. 
  - `data/vocabulary.txt`: vocabulary, frequent substructures. 

### data cleaning  
We remove the molecules that contains substructure that is not in vocabulary. 

```bash 
python clean.py 
```

- input 
  - `data/vocabulary.txt`: vocabulary 
  - `data/zinc.tab`: all the smiles in ZINC
- output
  - `data/zinc_clean.txt`

### Labelling
We use oracle to evaluate molecule's properties to obtain the labels for training graph neural network. 
```bash
python labelling.py
```
- input
  - `data/zinc_clean.txt`: all the smiles in ZINC, around 250K. 
- output
  - `data/zinc_label.txt`: including 6 columns, `smiles`, `qed`, `sa`, `jnk`, `gsk`, `logp`. We only contains subset of zinc (10K). 



## 3. Learning and Inference 

In our setup, we restrict the number of oracle calls in both training GNN and de novo design. 

### train graph neural network (GNN)

It corresponds to Section 3.2 in the paper. 
```bash 
python train.py $prop $train_oracle
```
- `prop` represent the property to optimize, including `qed`, `logp`, `jnk`, `gsk`, `jnkgsk`, `qedsajnkgsk`.  
- `train_oracle` is number of oracle calls in training GNN. 
- input 
  - `data/zinc_label.txt`: **training data** includes `(SMILES,y)` pairs, where `SMILES` is the molecule, `y` is the label. `y = GNN(SMILES)`
- output 
  - `pretrained_model/model_epoch_*.ckpt`: saved GNN model. 
- log
  - `"loss/{$prop}.pkl"` save the valid loss. 
For example, 
```bash 
python train.py jnkgsk 5000 
```




### de novo molecule design 




```bash
python run.py 
```
- input 
  - `pretrained_model/{$prop}_*.ckpt`: saved GNN model. * is number of iteration or epochs. 













