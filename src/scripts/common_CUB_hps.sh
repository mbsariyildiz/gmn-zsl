# sets the hyper-parameters common among all models for CUB
export FEATURE_NORM="none"

# network params
export D_NOISE=512
export N_G_HLAYER=1
export N_D_HLAYER=1
export N_G_HUNIT=2048
export N_D_HUNIT=1024
export DP_G=0.0
export DP_D=0.0
export LEAKINESS_G=0.2
export LEAKINESS_D=0.2
export GEN_TYPE='latent_noise'

# trainer params
export BATCH_SIZE=256
export GAN_OPTIM_LR_G=0.0001
export GAN_OPTIM_LR_D=0.0001
export GAN_OPTIM_BETA1=0.0
export GAN_OPTIM_BETA2=0.999
export N_D_ITER=5
export L=10.
export N_CKPT=2
export N_ITER=100000
export SEED=42

# classifier params
export CLF_TYPE='bilinear-comp'
export CLF_N_HLAYER=0
export CLF_N_HUNIT=0
export CLF_OPTIM_TYPE='sgd'
export CLF_LR=0.01
export CLF_LR_DECAY=0.97
export CLF_WD=0.001
export CLF_UNIFORM_SAMPLING=0
export CLF_N_EPOCH=200
export CLF_BATCH_SIZE=256
export N_SYNTH_S=0
export N_SYNTH_U=250

