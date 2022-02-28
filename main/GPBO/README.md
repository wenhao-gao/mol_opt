# Code for "A Fresh Look at De Novo Molecular Design Benchmarks"

This is the code for the [workshop paper](https://openreview.net/forum?id=gS3XMun4cl_) presented at NeurIPS 2021 AI for Science: Mind the Gaps workshop.
The code is fairly rough as it was originally part of a code base for another project.
I have tried to extract out just the parts for this workshop paper.
If you are unable to get the code to run, feel free to contact us.

## Data

All datasets are included in the repo.

## Python environment

- python 3.7
- scipy
- numpy
- pandas
- rdkit
- scikit-learn
- joblib
- gpytorch and botorch (for Gaussian process regression and BO)
- matplotlib
- networkx 

## Reproducing experiments

To reproduce the experiments, simply activate the conda environment, and run the scripts in the `experiments/ai4sci-sep2021` directory.
For example: `bash experiments/ai4sci-sep2021/graph_ga_v1.sh`


```bash
PYTHONPATH="$(pwd)/src:$PYTHONPATH" python run.py \
    --num_cpu=6 \
    \
    --dataset="./data/guacamol-splits/seed_1/rand_100.tsv" \
    --objective=QED \
    --maximize \
    --max_func_calls=1000 \
    \
    --n_train_gp_best=2000 \
    --n_train_gp_rand=3000 \
    --bo_batch_size=1000 \
    \
    --fp_radius=2 \
    --fp_nbits=4096 \
    \
    --ga_max_generations=5 \
    --ga_offspring_size=100 \
    --ga_pop_params 25 50 100 \
    \
    --output_path=result.json 
```



```bash
python run.py 
```
