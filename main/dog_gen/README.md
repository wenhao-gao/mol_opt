# DoG_Gen

This is the hill-climbing version of DoG. We simplify the [original version](https://github.com/john-bradshaw/synthesis-dags) and adapt it our environment. DoG is proposed in the paper "Barking up the right tree: an approach to search over molecule synthesis DAGs"
 by John Bradshaw, Brooks Paige, Matt J. Kusner, Marwin H. S. Segler, José Miguel Hernández-Lobato (https://arxiv.org/abs/2012.11522).
 

## Install (same with DoG_AE)

We need to install two conda environments, 
- 1. running Molecular Transformer to measure the uncertainty in chemical reaction prediction. 
- 2. runing main `DoG_Gen` program, under our package. 



### install 1: Molecular Transformer

Setup the Molecular Transformer<sup>[1](#refMolTran)</sup> as a server using [Schwaller et al.'s code](https://github.com/pschwllr/MolecularTransformer):  
    a. Open up a new shell (leave the old one open -- we will come back to it in step 3).
    b. In this new shell clone the Transformer repo somewhere, e.g. `git clone git@github.com:pschwllr/MolecularTransformer.git`  
    c. Install the relevant Python packages via installing an appropriate Conda environment, e.g. `conda env create -f conda_mtransformer_gpu.yml`   
    d. Activate the conda environment, e.g. `conda activate mtransformer_py3.6_pt0.4` (if you download the conda environment from this repo) 
    e. Add the weights to the Transformer directory (wherever you cloned it) inside a `saved_models` subdirectory. These weights
    can be downloaded from    
     [Google Drive](https://drive.google.com/file/d/1ogXzAg71BOs9SBrVt-umgcdc1_0ijUvU/view?usp=sharing):  
         ```shasum -a 256 molecular_transformer_weights.pt
        ## returns 93199b61da0a0f864e1d37a8a80a44f0ca9455645e291692e89d5405e786b450  molecular_transformer_weights.pt```  
     f. Inside the `available_models` subdirectory of the Transformer repo copy the `misc/mtransformer_example_server.conf.json` 
     file from this repo (you can change these parameters as you wish) into the Molecular Transformer repo.  
     g. From the top level of the Transformer repo start the server, with e.g. `CUDA_VISIBLE_DEVICES="0,1" python server.py --config available_models/mtransformer_example_server.conf.json`  
        (this is where you can choose which GPUs on your machine you want to use so edit the `CUDA_VISIBLE_DEVICES` variable appropriately).

    I assume you're running the Transformer on the same machine as our code, if not you'll want to edit 
    `synthesis-dags-config.ini` such that our code can find the Transformer server.
    Now just leave this server running in this shell and the code in this repo will communicate with it when necessary.   


### install 2: main `DoG_Gen` program

We have several steps as following:

    a. Install the conda environment: `conda env create -f conda_dogae_gpu.yml`  
    b. Activate it: `conda activate dogae_py3.7_pt1.4`  
    d. Make sure you have cloned the submodules of this repo, i.e. `git submodule init` and `git submodule update`  



## Data 


Unzip the `uspto.zip` (47M) folder in this folder and in `scripts/dataset_creation/data.zip` (5.3M).


`scripts/dataset_creation/data/uspto-train-depth_and_tree_tuples.pick` is used for run. 

## Pretrain 

Run `mol_opt/main/dog_gen/scripts/dog_gen/train_dog_gen.py` 


Pretrained DoG-Gen model is saved in `mol_opt/main/dog_gen/scripts/dog_gen/chkpts/doggen_weights.pth.pick`. 


## Run 

### 1. Molecular Transformer

Open a new shell, and run the following code and leave it open: 
```bash
conda activate mtransformer_py3.6_pt0.4
cd MolecularTransformer 
CUDA_VISIBLE_DEVICES="0,1" python server.py --port 5001 --config available_models/mtransformer_example_server.conf.json
```

### 2. main `DoG_Gen` program

Make sure the port number in `main/dog_ae/synthesis-dags-config.ini` is the same as `5001` in Molecular Transformer above. 


Open a new shell, run the following codes and wait the results

```bash
conda activate dogae_py3.7_pt1.4
cd mol_opt 
python run.py dog_gen 
```



 

 
