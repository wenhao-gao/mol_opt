#!/usr/bin/env bash
# You should source this script to get the correct additions to Python path

# Directory of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up the python paths
export PYTHONPATH=${PYTHONPATH}:${DIR}
export PYTHONPATH=${PYTHONPATH}:${DIR}/submodules/autoencoders
export PYTHONPATH=${PYTHONPATH}:${DIR}/submodules/GNN

