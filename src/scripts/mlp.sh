# hyper-parameters for CUB
# mlp, SGD with momentum=0.9, lr=1e-2, wd=1e-3, n_epochs=200
# - zsl   = 
# - gzslu = 
# - gzsls = 73.0 += 0.4
# - gzslh = 

# hyper-parameters for SUN
# mlp, SGD with momentum=0.9, lr=1e-0, lr_decay=0.96, wd=1e-2, n_epochs=200
# - zsl   = 
# - gzslu = 
# - gzsls = 49.6 +- 0.2
# - gzslh = 

# hyper-parameters for AWA1
# mlp, SGD with momentum=0.9, lr=1e-2, wd=1e-3, n_epochs=200
# - zsl   = 
# - gzslu = 
# - gzsls = 91.3 += 0.1
# - gzslh = 

export CODE_DIR="" # required
export MODE='test'
export CLF_TYPE='mlp'
export CLF_N_HLAYER=0
export CLF_N_HUNIT=0
export CLF_UNIFORM_SAMPLING=0
export CLF_N_EPOCH=200
export CLF_BATCH_SIZE=256
export FEATURE_NORM="none"

export DATASET='AWA1'
export DATA_DIR=../data/${DATASET}
export CLF_OPTIM_TYPE='sgd'
export CLF_LR=1.0
export CLF_LR_DECAY=0.97
export CLF_WD=0.01
export CLF_EXP_ROOT=../experiments/${DATASET}/${MODE}/clf-models/final-${CLF_TYPE}
bash scripts/run_classifier.sh

