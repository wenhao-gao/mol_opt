# genetic_expert_guided_learning

This is an official implementation of our paper Guiding Deep Molecular Optimization with Genetic Exploration (https://arxiv.org/pdf/2007.04897.pdf). Our code is largely inspired by GuacaMol baselines (https://github.com/BenevolentAI/guacamol_baselines).

## 1. Setting up the environment
You can set up the environment by following commands. dmo is shortcut for deep-molecular-optimization
```
conda create -n dmo python=3.6
conda activate dmo
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c dglteam dgl-cuda10.1
conda install -c rdkit rdkit
pip install neptune-client
pip install tqdm
pip install psutil
pip install guacamol
```

You also need to get a (free) neptune account and modify the project_qualified_name variable for neptune initialization (in our files run_pretrain.py, run_gegl.py, and run_gegl_constrained.py):

```
neptune.init(project_qualified_name="sungsoo.ahn/deep-molecular-optimization")
```



## 2. Dataset and pretrained models
Note: you can skip this part since we already provide the dataset and the pretrained models in this repository.

### 2.1. ZINC 250k
For Table 1, we download the ZINC dataset from the official implementation of Junction Tree VAE as follows:

```
wget https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/all.txt ./resource/data/zinc/all.txt
wget https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/train.txt ./resource/data/zinc/train.txt
wget https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/valid.txt ./resource/data/zinc/valid.txt
wget https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/test.txt ./resource/data/zinc/test.txt
```

Then we pretrain our neural apprentice policy on the ZINC training dataset.

```
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
--dataset zinc \
--dataset_path ./resource/data/zinc/train.txt \
--save_dir ./resource/checkpoint/zinc/
```

Then we hard-code the normalization constants for the penalized logP score in util/chemistry/benchmarks.py file. To obtain the values, we compute the normalization statistics for the penalized logP as follows:
```
python get_logp_stats.py
```

Finally, for Table 2(b), we record the lowest scoring molecules in a separate file as follows:
```
python get_low_scoring_dataset.py
```

### 2.2. GuacaMol
We use the neural apprentic policy pretrained on the GuacaMol dataset, provided by [Brown et al. 2019].

```
wget https://github.com/BenevolentAI/guacamol_baselines/blob/master/smiles_lstm_hc/pretrained_model/model_final_0.473.pt ./resource/checkpoint/guacamol/generator_weight.pt
wget https://github.com/BenevolentAI/guacamol_baselines/blob/master/smiles_lstm_hc/pretrained_model/model_final_0.473.pt ./resource/checkpoint/guacamol/generator_config.json
```

## 3. Unconstrained optimization of penalized logP

We use the following command to obtain the results in Table 1(a) of our paper.
```
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python run_gegl.py\
--benchmark_id 28 \
--dataset zinc \
--apprentice_load_dir ./resource/checkpoint/zinc \
--max_smiles_length 81
```

## 4. Constrained optimization of penalized logP
We use the following command to obtain the results in Table 1(b) of our paper.

```
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python run_gegl_constrained.py \
--dataset zinc \
--dataset_path ./resource/data/zinc/logp_800.txt \
--apprentice_load_dir ./resource/checkpoint/zinc \
--similarity_threshold 0.4 \
--smi_id_min 0 --smi_id_max 800
```

## 5. GuacaMol benchmark
We use the following command to obtain the results in Table 2 of our paper. The variable BENCHMARK_ID ranges between 0, ..., 19.

```
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python run_gegl.py \
--benchmark_id $BENCHMARK_ID \
--dataset guacamol
--apprentice_load_dir ./resource/checkpoint/guacamol
--record_filtered
```