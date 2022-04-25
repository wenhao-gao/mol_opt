# Junction Tree Variational Autoencoder 

# data 

```bash
export PYTHONPATH=$PREFIX/JTVAE
```

## generate vocabulary 

```bash
cd JTVAE/fast_molvae 
python ../fast_jtnn/mol_tree.py < ../data/moses/train.txt 
```
It take a long time 


## Preprocess 
```bash
cd JTVAE/fast_molvae 
python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16
mkdir moses-processed
mv tensor* moses-processed
```


# train VAE (trained)

```bash 
mkdir JTVAE/vae_model/
python vae_train.py --train moses-processed --vocab ../data/moses/vocab.txt --save_dir vae_model/
```

- `data`: `../data/moses/train.txt`
- `output`: trained VAE model `../fast_molvae/vae_model/model.iter-xxxx`. For efficiency, we save a well-trained VAE. 


# BO (starting point)

Users can start from BO for efficiency. 

```bash
source activate molopt 
```


```bash 
cd JTVAE/bo
export PYTHONPATH=/project/molecular_data/graphnn/mol_opt/main/JTVAE/ 
```

```bash
python preprocess.py 
```


```bash 
mkdir results
python bo.py --vocab ../data/moses/vocab.txt --save_dir results --data ../data/moses/train_validity_5k.txt --hidden 450 --latent 56 --model ../fast_molvae/vae_model/model.iter-5000
```













