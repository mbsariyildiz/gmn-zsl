#!/bin/bash

export DATASET='AWA1'
export MODE='test'
export CODE_DIR="." 
export DATA_DIR="" # required
export EXP_ROOT="./experiments/${DATASET}/${MODE}/gmn"

# set common params
source ${CODE_DIR}/scripts/common_AWA1_hps.sh

# gmn trainer params
export Q=100.
export Z=0.
export CLF_RESET_ITER=1
export N_GM_ITER=2
export PER_CLASS_BATCH_SIZE=256
export GM_FAKE_REPEAT=2
export EXP_NO=1

bash ${CODE_DIR}/scripts/run_main.sh
