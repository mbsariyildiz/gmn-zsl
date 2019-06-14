export DATASET='AWA1'
export MODE='test'
export CODE_DIR="" # required
export DATA_DIR=../data/${DATASET}
export EXP_ROOT=../experiments/${DATASET}/${MODE}/cwgan

# set common params
source ${CODE_DIR}/scripts/common_AWA1_hps.sh

export EXP_NO=1
bash ${CODE_DIR}/scripts/run_main.sh

