#!/bin/bash

export DATASET='CUB'
export MODE='test'
export CODE_DIR="" # required
export DATA_DIR=../data/${DATASET}
export EXP_ROOT=../experiments/${DATASET}/${MODE}/cwgan

# regg trainer params
export REGG_LR=0.000099978 # taken from Felix et.al.
export REGG_WD=0.0         # https://github.com/rfelixmg/frwgan-eccv18/blob/master/src/sota/cub/architecture/cycle_wgan.json
export REGG_N_EPOCH=40     #
export REGG_BATCH_SIZE=64  #
export REGG_EXP_ROOT=${EXP_ROOT}/final

bash ${CODE_DIR}/scripts/run_regressor.sh

