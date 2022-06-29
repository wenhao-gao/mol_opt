#!/bin/bash

# methods=('screening' 'molpal' 'graph_ga' 'smiles_ga' 'selfies_ga' \
#          'graph_mcts' 'smiles_lstm_hc' 'selfies_lstm_hc' 'gpbo' 'jt_vae' 'moldqn' \
#          'smiles_vae' 'selfies_vae' 'chembo' 'mars' 'reinvent' 'reinvent_selfies')
methods=('stoned' 'pasithea' 'gflownet' 'gflownet_al' 'mimosa')

for method in "${methods[@]}"
do
    echo "#!/bin/bash

oracle_array=('jnk3' 'gsk3b' 'celecoxib_rediscovery' \\
    'troglitazone_rediscovery' \\
    'thiothixene_rediscovery' 'albuterol_similarity')

for oralce in "\${oracle_array[@]}"
do
python -u run.py ${method} --task production --n_runs 5 --n_jobs 8 --wandb offline --max_oracle_calls 10000 --oracles \${oralce}
done" > production1_$method.slurm




    echo "#!/bin/bash

oracle_array=('mestranol_similarity' \\
    'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo')

for oralce in "\${oracle_array[@]}"
do
python -u run.py ${method} --task production --n_runs 5 --n_jobs 8 --wandb offline --max_oracle_calls 10000 --oracles \${oralce}
done" > production2_$method.slurm




    echo "#!/bin/bash

oracle_array=('fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \\
    'sitagliptin_mpo' 'zaleplon_mpo')

for oralce in "\${oracle_array[@]}"
do
python -u run.py ${method} --task production --n_runs 5 --n_jobs 8 --wandb offline --max_oracle_calls 10000 --oracles \${oralce}
done" > production3_$method.slurm





    echo "#!/bin/bash

oracle_array=('valsartan_smarts' 'deco_hop' 'scaffold_hop' 'qed' 'drd2')

for oralce in "\${oracle_array[@]}"
do
python -u run.py ${method} --task production --n_runs 5 --n_jobs 8 --wandb offline --max_oracle_calls 10000 --oracles \${oralce}
done" > production4_$method.slurm

#     if [[ ${method} = 'molpal' ]]
#     then
#         CUDA_VISIBLE_DEVICES=0 nohup bash production_${method}.sh &> ${method}_prod.out &
#     elif [[ $method = 'smiles_lstm_hc' ]]
#     then
#         CUDA_VISIBLE_DEVICES=1 nohup bash production_${method}.sh &> ${method}_prod.out &
#     elif [[ $method = 'selfies_lstm_hc' ]]
#     then
#         CUDA_VISIBLE_DEVICES=2 nohup bash production_${method}.sh &> ${method}_prod.out &
#     elif [[ $method = 'selfies_ga' ]]
#     then
#         CUDA_VISIBLE_DEVICES=3 nohup bash production_${method}.sh &> ${method}_prod.out &
#     else
#         CUDA_VISIBLE_DEVICES= nohup bash production_${method}.sh &> ${method}_prod.out &
#     fi
done
