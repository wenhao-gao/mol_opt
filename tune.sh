#!/bin/bash 

nohup python -u main/graph_mcts/run.py --n_jobs 16 --task tune --n_runs 50 --oracles zaleplon_mpo perindopril_mpo > tune.out 2>&1 &
