#!/bin/bash

export DATASET='CUB'
export MODE='test'
export CODE_DIR="" # required
export DATA_DIR=../data/${DATASET}
export EXP_ROOT=../experiments/${DATASET}/${MODE}/cwgan

# set common params
source scripts/common_CUB_hps.sh

# fcls-wgan params
export PRETRAINED_CLF_CKPT=../experiments/${DATASET}/${MODE}/clf-models/final-bilinear-comp/clf-model_000/model.ckpt
export C=0.01
export EXP_NO=1

bash scripts/run_main.sh

