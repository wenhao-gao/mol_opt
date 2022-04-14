#!/bin/bash 

model="gpbo"

nohup python -u run.py ${model} \
    --n_jobs 16 --task tune --n_runs 30 \
    --oracles zaleplon_mpo perindopril_mpo > tune_${model}.out 2>&1 &

