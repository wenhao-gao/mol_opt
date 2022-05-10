#!/bin/bash 

model="mars"

nohup python -u run.py ${model} \
    --n_jobs 8 --task tune --n_runs 30 --wandb online \
    --oracles zaleplon_mpo perindopril_mpo > tune_${model}.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0
