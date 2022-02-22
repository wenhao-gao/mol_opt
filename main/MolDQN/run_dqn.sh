#!/bin/bash

#TASK=`date +"%Y%m%d_%T"`_synthdqn
#HPARAMS='agents/configs/test.json'
#
#source activate rdkit
#CUDA_VISIBLE_DEVICES=0 nohup python -u run_dqn.py \
#    -t ${TASK} \
#    -c ${HPARAMS} \
#    --synthesizability sa \
#    -o logp qed &> ${TASK}.out&

TASK='test'
HPARAMS='agents/configs/test.json'
CUDA_VISIBLE_DEVICES=0 python -u run_dqn.py \
    -t ${TASK} \
    -c ${HPARAMS} \
    -o qed \
    --synthesizability smi \
    --q_function mlp
