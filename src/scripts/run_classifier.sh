
[[ -z "${GEN_TYPE}" ]] && GEN_TYPE=""
[[ -z "${D_NOISE}" ]] && D_NOISE=0
[[ -z "${N_G_HLAYER}" ]] && N_G_HLAYER=0
[[ -z "${N_G_HUNIT}" ]] && N_G_HUNIT=0
[[ -z "${LEAKINESS_G}" ]] && LEAKINESS_G=0.2
[[ -z "${DP_G}" ]] && DP_G=0.0
[[ -z "${N_SYNTH_U}" ]] && N_SYNTH_U=0
[[ -z "${N_SYNTH_S}" ]] && N_SYNTH_S=0
[[ -z "${DEVICE}" ]] && DEVICE="cuda"

python ${CODE_DIR}/train_classifier.py \
	--dataset=$DATASET \
	--feature_norm=$FEATURE_NORM \
	--data_dir=$DATA_DIR \
	--exp_root=$CLF_EXP_ROOT \
	--mode=$MODE \
	--device=$DEVICE \
    --clf_type=$CLF_TYPE \
    --clf_n_hlayer=$CLF_N_HLAYER \
    --clf_n_hunit=$CLF_N_HUNIT \
    --clf_optim_type=$CLF_OPTIM_TYPE \
    --clf_lr=$CLF_LR \
    --clf_lr_decay=$CLF_LR_DECAY \
    --clf_wd=$CLF_WD \
	--uniform_sampling=$CLF_UNIFORM_SAMPLING \
	--n_epoch=$CLF_N_EPOCH \
	--batch_size=$CLF_BATCH_SIZE \
	--generator_ckpt=$GENERATOR_CKPT \
	--gen_type=$GEN_TYPE \
	--d_noise=$D_NOISE \
	--n_g_hlayer=$N_G_HLAYER \
	--n_g_hunit=$N_G_HUNIT \
    --leakiness_g=$LEAKINESS_G \
    --dp_g=$DP_G \
	--n_synth_U=$N_SYNTH_U \
	--n_synth_S=$N_SYNTH_S 
