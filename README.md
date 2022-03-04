# mol_opt



## install 

```bash
conda create -n molopt python=3.7
conda activate molopt 
conda install -c rdkit rdkit 
pip install torch 
pip install guacamol 
pip install PyTDC 
pip install networkx 
pip install joblib 
pip install nltk 
```

```bash
conda activate molopt 
```

## step by step  

- runable 
- make compatible (run.py with TDC type oracle)
- hyper-param tuning 
- test run 
- code unify 

## Models

|                    | `runable` | `compatible` | `hparam` | `test` | `unity` |
|--------------------|-----------|--------------|----------|--------|---------|
| **Screening**      | ✅         | ✅           | ✅       |        |         |
| **Graph Ga**       | ✅         | ✅           | ✅       |        |         |
| **SMILES GA**      | ✅         |              |          |        |         |
| **SELFIES GA**     | ✅         |              |          |        |         |
| **SELFIES GA +D**  | ✅         |              |          |        |         |
| **Graph MCTS**     | ✅         |              |          |        |         |
| **SMILES HC**      |           |              |          |        |         |
| **SELFIES HC**     |           |              |          |        |         |
| **SMILES VAE BO**  |           |              |          |        |         |
| **SELFIES VAE BO** |           |              |          |        |         |
| **JTVAE BO**       | ✅        |              |          |        |         |
| **BOSS (SMILES)**  |           |              |          |        |         |
| **BOSS (SELFIES)** |           |              |          |        |         |
| **Graph-GA+GP-BO** |           |              |          |        |         |
| **ORGAN**          | doing     |              |          |        |         |
| **MolGAN**         | doing     |              |          |        |         |
| **ChemBO**         | -         |              |          |        |         |
| **REINVENT**       |           |              |          |        |         |
| **RationaleRL**    |           |              |          |        |         |
| **MolDQN**         | ✅        |              |          |        |         |
| **MIMOSA**         | ✅        |              |          |        |         |
| **MARS**           | doing     |              |          |        |         |
| **DoG-Gen**        | -         |              |          |        |         |
| **DoG-AE BO**      | -         |              |          |        |         |
| **SynNet**         |           |              |          |        |         |
| **Pasithea**       |           |              |          |        |         |
| **DST**            | ✅        |              |          |        |         |
| **GFlowNet**       |           |              |          |        |         |
| **GFlowNet (AL)**  |           |              |          |        |         |

## Run 

First run the following to add current path to python path:
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### graph-GA

```bash
python main/graph_ga/run.py
```

### smiles-GA 

```bash 
python main/smiles_ga/run.py 
```


### graph-mcts 

```bash
python main/graph_mcts/run.py 
``` 


### selfies-GA 

```bash
pip install selfies 
pip install tensorboardX 
```


```bash
python main/selfies_GA/run.py 
```


### DST 


```bash
python main/DST/run.py 
```


### MolDQN 

```bash
python main/MolDQN/run.py 
```




### JTVAE 

```bash
python main/JTVAE/run.py 
```



### GP-BO 


```bash 
python main/GP_BO/run.py 
```











