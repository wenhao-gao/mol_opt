#!/usr/bin/env bash

source /opt/conda/etc/profile.d/conda.sh
source ./docker_set_up.sh
cd testing
pytest
