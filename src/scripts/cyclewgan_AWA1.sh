#!/bin/bash

export DATASET='AWA1'
export MODE='test'
export CODE_DIR="" # required
export DATA_DIR=../data/${DATASET}
export EXP_ROOT=../experiments/${DATASET}/${MODE}/cwgan

# set common params
source scripts/common_AWA1_hps.sh

# regressor params
export REGG_N_HLAYER=0
export REGG_N_HUNIT=0
export PRETRAINED_REGG_CKPT=../experiments/${DATASET}/${MODE}/regg/final/regg-model_000/model.ckpt
export R=0.01

export EXP_NO=1
bash scripts/run_main.sh

