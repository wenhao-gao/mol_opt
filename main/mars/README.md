# MARS: Markov Molecular Sampling for Multi-objective Drug Discovery


## Dependencies

```bash
conda install tqdm tensorboard scikit-learn
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c dglteam dgl-cuda11.1

# for cpu only
conda install pytorch cpuonly -c pytorch
conda install -c dglteam dgl
```


## preprocess

```bash
python datasets/prepro_vocab.py 
```

## Run 

```bash
python run.py 
```


