#!/bin/bash
# 1/1, nU=175 gives:
# - zsl   = 61.1
# - gzslu = 50.3 
# - gzsls = 37.2
# - gzslh = 42.8

export DATASET='SUN'
export MODE='test'
export CODE_DIR="." 
export DATA_DIR="" # required
export EXP_ROOT="./experiments/${DATASET}/${MODE}/gmn"

# set common params
source ${CODE_DIR}/scripts/common_SUN_hps.sh

# gmn trainer params
export Q=100.
export Z=0.
export CLF_RESET_ITER=1
export N_GM_ITER=20
export PER_CLASS_BATCH_SIZE=16
export GM_FAKE_REPEAT=10
export N_ITER=20000
export EXP_NO=1
bash ${CODE_DIR}/scripts/run_main.sh


