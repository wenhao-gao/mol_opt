#!/bin/bash

TASK=`date +"%Y%m%d_%T"`_dqn_zinc
HPARAMS='agents/configs/zinc_space.json'

# CUDA_VISIBLE_DEVICES=0 nohup python -u run_dqn.py \
#     -t ${TASK} \
#     --path_to_config ${HPARAMS} \
#     -o logp qed mpo_zaleplon 


CUDA_VISIBLE_DEVICES=0 python run_dqn.py \
    -t ${TASK} \
    --path_to_config ${HPARAMS} 



### problematic 
# CUDA_VISIBLE_DEVICES=0 nohup python -u run_dqn.py \
#     -t ${TASK} \
#     --hparams ${HPARAMS} \
#     -o logp qed mpo_zaleplon 



    # &> ${TASK}.out&
#CUDA_VISIBLE_DEVICES=0 python -u run_dqn.py \
#    -t ${TASK} \
#    --hparams ${HPARAMS} \
#    -o logp qed mpo_zaleplon
