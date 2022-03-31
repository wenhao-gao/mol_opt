#!/bin/bash 

export PYTHONPATH=`pwd`

model="graph_ga"

nohup python -u main/${model}/run.py \
    --n_jobs 16 --task tune --n_runs 30 \
    --oracles zaleplon_mpo perindopril_mpo > tune_${model}.out 2>&1 &

