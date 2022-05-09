# MolGAN
Tensorflow implementation of MolGAN: An implicit generative model for small molecular graphs (https://arxiv.org/abs/1805.11973)

## Dependencies

* **python>=3.6**
* **tensorflow>=1.7.0**

## Structure
* [data](https://github.com/nicola-decao/MolGAN/tree/master/data): should contain your datasets. If you run `download_dataset.sh` the script will download the dataset used for the paper (then you should run `python utils/sparse_molecular_dataset.py` to convert the dataset in a graph format used by MolGAN models).
* [example](https://github.com/nicola-decao/MolGAN/blob/master/example.py): Example code for using the library within a Tensorflow project. **NOTE: these are NOT the experiments on the paper!**
* [models](https://github.com/nicola-decao/MolGAN/tree/master/models): Class for Models. Both VAE and (W)GAN are implemented.
* [optimizers](https://github.com/nicola-decao/MolGAN/tree/master/optimizers): Class for Optimizers for both VAE, (W)GAN and RL.

## run 

```bash
python run.py 
```


