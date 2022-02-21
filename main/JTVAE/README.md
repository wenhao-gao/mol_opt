# Junction Tree Variational Autoencoder for Molecular Graph Generation

Official implementation of our Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)



# Accelerated Version
We have accelerated our code! The new code is in `fast_jtnn/`, and the VAE training script is in `fast_molvae/`. Please refer to `fast_molvae/README.md` for details.

# Requirements
* RDKit (version >= 2017.09)
* Python (version == 2.7)
* PyTorch (version >= 0.2)


We highly recommend you to use conda for package management.

# Quick Start
The following directories contains the most up-to-date implementations of our model:
* `fast_jtnn/` contains codes for model implementation.
* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.

The following directories provides scripts for the experiments in our original ICML paper:
* `bo/` includes scripts for Bayesian optimization experiments. Please read `bo/README.md` for details.
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training our VAE model.
* `molopt/` includes scripts for jointly training our VAE and property predictors. Please read `molopt/README.md` for details.
* `jtnn/` contains codes for model formulation.



# data 

```bash
export PYTHONPATH=$PREFIX/JTVAE
export PYTHONPATH=/project/molecular_data/graphnn/mol_opt/main/JTVAE/
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


# train VAE

```bash 
mkdir JTVAE/vae_model/
python vae_train.py --train moses-processed --vocab ../data/moses/vocab.txt --save_dir vae_model/
```

- `data`: 
- `output`: trained VAE model `../fast_molvae/vae_model/model.iter-xxxx`. For efficiency, user can download it. 


# BO 

```bash 
cd JTVAE/bo
```

## generate latent representation of all training molecules

```bash
python gen_latent.py --data ../data/moses/train_validity_5k.txt --vocab ../data/moses/vocab.txt \
--hidden 450 --latent 56 \
--model ../fast_molvae/vae_model/model.iter-5000
```



## run BO 
```bash 
mkdir results
python run_bo.py --vocab ../data/moses/vocab.txt --save_dir results \
--hidden 450 --latent 56 --model ../fast_molvae/vae_model/model.iter-5000
```



## whole pipeline
```bash 
mkdir results
python bo.py --vocab ../data/moses/vocab.txt --save_dir results --data ../data/moses/train_validity_5k.txt \
--hidden 450 --latent 56 --model ../fast_molvae/vae_model/model.iter-5000
```



