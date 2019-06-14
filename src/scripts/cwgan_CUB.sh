#!/bin/bash

export DATASET='CUB'
export MODE='test'
export CODE_DIR="" # required
export DATA_DIR=../data/${DATASET}
export EXP_ROOT=../experiments/${DATASET}/${MODE}/cwgan

# set common params
source scripts/common_CUB_hps.sh

export EXP_NO=1
bash scripts/run_main.sh

