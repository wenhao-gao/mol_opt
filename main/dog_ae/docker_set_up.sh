#!/usr/bin/env bash
# This script should be the entrypoint to Docker so that the Transformer is started up automatically and PythonPath is correct
# etc. Should be sourced so that can change environment variables.

# Set up a Molecular Transformer to run in the background.
cd /molecular_transformer
conda activate mtransformer_py3.6_pt0.4
CUDA_VISIBLE_DEVICES="0,1" python server.py --config available_models/mtransformer_example_server.conf.json &

# Activate conda environment for the code in this repo and then add items to Python path.
cd /synthesis-dags
conda activate dogae_py3.7_pt1.4
source set_up.sh

