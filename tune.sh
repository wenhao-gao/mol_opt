#!/bin/bash 

model="boss"

nohup python -u run.py ${model} \
    --n_jobs 16 --task tune --max_oracle_calls 10000 --wandb online \
    --oracles zaleplon_mpo > tune_${model}.out 2>&1 &
    # --n_jobs 16 --task tune --n_runs 30 \
    # --oracles zaleplon_mpo perindopril_mpo > tune_${model}.out 2>&1 &

