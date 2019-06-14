# hyper-parameters for CUB
# bilinear comp SGD with momentum=0.9, lr=1e-2, wd=1e-3, n_epochs=200
# - zsl   = 57.8 +- 0.1
# - gzslu = 27.5 +- 0.2
# - gzsls = 66.9 +- 0.1
# - gzslh = 39.0 +- 0.1

# hyper-parameters for SUN
# bilinear-comp, SGD with momentum=0.9, lr=1e-2, wd=1e-3, n_epochs=200
# - zsl   = 61.5 +- 0.1
# - gzslu = 25.0 +- 0.3
# - gzsls = 40.3 +- 0.3
# - gzslh = 30.9 +- 0.3

# hyper-parameters for AWA1
# bilinear-comp, SGD with momentum=0.9, lr=1e-3, wd=1e-1, n_epochs=200
# - zsl   = 63.2 +- 0.1
# - gzslu = 13.4 +- 0.3
# - gzsls = 78.2 +- 0.3
# - gzslh = 22.9 +- 0.3

export CODE_DIR="" # required
export MODE='test'
export CLF_TYPE='bilinear-comp'
export CLF_N_HLAYER=0
export CLF_N_HUNIT=0
export CLF_OPTIM_TYPE='adam'
export CLF_LR_DECAY=0.96
export CLF_UNIFORM_SAMPLING=0
export CLF_N_EPOCH=200
export CLF_BATCH_SIZE=256
export FEATURE_NORM="none"

export DATASET='SUN'
export DATA_DIR=../data/${DATASET}
export CLF_LR=0.01
export CLF_WD=0.001
export CLF_EXP_ROOT=../experiments/${DATASET}/${MODE}/clf-models/final-${CLF_TYPE}
bash scripts/run_classifier.sh

export DATASET='AWA1'
export MODE='test'
export DATA_DIR=../data/${DATASET}
export CLF_EXP_ROOT=../experiments/${DATASET}/${MODE}/clf-models/final-${CLF_TYPE}
export CLF_LR=0.1
export CLF_WD=0.0
bash scripts/run_classifier.sh

export DATASET='CUB'
export MODE='test'
export DATA_DIR=../data/${DATASET}
export CLF_EXP_ROOT=../experiments/${DATASET}/${MODE}/clf-models/final-${CLF_TYPE}
export CLF_LR=0.001
export CLF_WD=0.0
bash scripts/run_classifier.sh

