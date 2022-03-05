#!/bin/bash 

nohup python -u main/smiles_ga/run.py --n_jobs 16 --task tune --n_runs 50 --oracles zaleplon_mpo perindopril_mpo > tune_smiles_ga.out 2>&1 &
