#!/bin/bash  

export PYTHONPATH=`pwd`

ulimit -n 4096

METHOD='DST'
OBJ='zaleplon_mpo'
# OBJ='ranolazine_mpo'

CUDA_VISIBLE_DEVICES= nohup python -u main/${METHOD}/run.py \
    --task production \
    --n_runs 5 \
    --oracles ${OBJ} &> ${METHOD}_${OBJ}.out&
