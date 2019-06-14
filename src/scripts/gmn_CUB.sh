#!/bin/bash
# 1/nU-0250_nS-0000 gives:
# zsl   : 67.0
# gzsl-u: 54.7
# gzsl-s: 58.4
# gzsl-h: 56.5

export DATASET='CUB'
export MODE='test'
export CODE_DIR="." 
export DATA_DIR="" # required
export EXP_ROOT="./experiments/${DATASET}/${MODE}/gmn"

# set common params
source ${CODE_DIR}/scripts/common_CUB_hps.sh

# gmn trainer params
export Q=100.
export Z=0.
export CLF_RESET_ITER=1
export N_GM_ITER=5
export PER_CLASS_BATCH_SIZE=60
export GM_FAKE_REPEAT=2
export EXP_NO=1
bash ${CODE_DIR}/scripts/run_main.sh

