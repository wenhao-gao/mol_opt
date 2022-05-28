# ChemBO

ChemBO is based on Dragonfly - a framework for scalable Bayesian optimization. We modify the Dragonfly, and attach it. 
The original README document is [here](https://github.com/ks-korovina/chembo). 

## setup conda environment

First, set up environment for RDKit and Dragonfly:

```bash
conda create -n chembo python=3.7
conda activate chembo
```

Install basic requirements with pip:

```bash
pip install -r requirements.txt
```

Install mol_opt environment: 
```bash
pip install torch 
pip install guacamol 
pip install PyTDC 
pip install networkx 
pip install joblib 
pip install nltk 
conda install -c rdkit rdkit 
```


Install Dragonfly (our modified version), it is available in this folder. 

```bash
cd mol_opt/chembo/dragonfly
python setup.py install 
```

We also attach `chembo.yml` for reference. 



### Getting data

```bash
cd mol_opt/chembo 
bash download_data.sh
```


## Running experiments


```bash
cd mol_opt 
python run.py chembo --oracles qed --seed 1 --wandb offline 
```









