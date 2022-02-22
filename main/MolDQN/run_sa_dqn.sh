#!/bin/bash

TASK=`date +"%Y%m%d_%T"`_dist_zaleplon
HPARAMS='agents/configs/zaleplon.json'
#TASK='test'
#HPARAMS='agents/configs/test.json'

source activate rdkit
CUDA_VISIBLE_DEVICES=3 nohup python -u run_dqn.py \
    -t ${TASK} \
    -c ${HPARAMS} \
    --synthesizability sa \
    --q_function mlp \
    -o mpo_zaleplon &> ${TASK}.out&
#CUDA_VISIBLE_DEVICES=0 python -u run_dqn.py \
#    -t ${TASK} \
#    -c ${HPARAMS} \
#    --synthesizability sa \
#    -o mpo_zaleplon
