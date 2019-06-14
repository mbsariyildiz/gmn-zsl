TRAIN_GEN_MODEL=1
TRAIN_CLF=1

[[ -z "${GAN_OPTIM_WD}" ]] && GAN_OPTIM_WD=0.0
[[ -z "${GAN_OPTIM_BETA1}" ]] && GAN_OPTIM_BETA1=0.9
[[ -z "${GAN_OPTIM_BETA2}" ]] && GAN_OPTIM_BETA2=0.9
[[ -z "${D_NORMALIZE_FT}" ]] && D_NORMALIZE_FT=0
[[ -z "${DP_G}" ]] && DP_G=0.0
[[ -z "${DP_D}" ]] && DP_D=0.0
[[ -z "${LEAKINESS_G}" ]] && LEAKINESS_G=0.2
[[ -z "${LEAKINESS_D}" ]] && LEAKINESS_D=0.2
[[ -z "${CLF_TYPE}" ]] && CLF_TYPE='bilinear-comp'
[[ -z "${CLF_OPTIM_TYPE}" ]] && CLF_OPTIM_TYPE='sgd'
[[ -z "${CLF_LR}" ]] && CLF_LR=0.0
[[ -z "${CLF_WD}" ]] && CLF_WD=0.0
[[ -z "${CLF_RESET_ITER}" ]] && CLF_RESET_ITER=-1
[[ -z "${N_GM_ITER}" ]] && N_GM_ITER=0
[[ -z "${PER_CLASS_BATCH_SIZE}" ]] && PER_CLASS_BATCH_SIZE=0
[[ -z "${GM_FAKE_REPEAT}" ]] && GM_FAKE_REPEAT=0
[[ -z "${Q}" ]] && Q=0.0
[[ -z "${Z}" ]] && Z=0.0
[[ -z "${C}" ]] && C=0.0
[[ -z "${R}" ]] && R=0.0
[[ -z "${PRETRAINED_CLF_CKPT}" ]] && PRETRAINED_CLF_CKPT=""
[[ -z "${PRETRAINED_REGG_CKPT}" ]] && PRETRAINED_REGG_CKPT=""
[[ -z "${N_CKPT}" ]] && N_CKPT=2
[[ -z "${N_ITER}" ]] && N_ITER=100000
[[ -z "${SEED}" ]] && SEED=42
[[ -z "${DEVICE}" ]] && DEVICE="cuda"

if [ "${DATASET}" == "CUB" ]; then
	N_SYNTH_SS=(0)
	N_SYNTH_US=(200 225 250 275 300 325 350 375 400)
elif [ "${DATASET}" == "SUN" ]; then
	N_SYNTH_SS=(0)
	N_SYNTH_US=(50 75 100 125 150 175 200 225 250 275 300 325 350 375 400)
elif [ "${DATASET}" == "AWA1" ]; then
	N_SYNTH_SS=(600)
	N_SYNTH_US=(2400 2300 2200 2100 2000 1900 1800 1700 1600 1500 1400 1300 1200 1100 1000)
fi

seeds=(1 67 99)

# for each seed
for six in $(seq 1 1 ${#seeds[@]}); do
	export SEED=${seeds[((six-1))]}
	export EXP_DIR=${EXP_ROOT}/$(printf "%03d/%d" $EXP_NO $six)
	export GENERATOR_CKPT=${EXP_DIR}/model.ckpt

	# train a generative model
    if [ $TRAIN_GEN_MODEL -eq 1 ]; then
        python ${CODE_DIR}/main.py \
            --dataset=$DATASET \
            --exp_dir=$EXP_DIR \
            --data_dir=$DATA_DIR \
            --mode=$MODE \
            --device=$DEVICE \
            --gen_type=$GEN_TYPE \
            --d_noise=$D_NOISE \
            --n_g_hlayer=$N_G_HLAYER \
            --n_g_hunit=$N_G_HUNIT \
            --dp_g=$DP_G \
            --leakiness_g=$LEAKINESS_G \
            --d_normalize_ft=$D_NORMALIZE_FT \
            --n_d_hlayer=$N_D_HLAYER \
            --n_d_hunit=$N_D_HUNIT \
            --dp_d=$DP_D \
            --leakiness_d=$LEAKINESS_D \
            --n_d_iter=$N_D_ITER \
            --L=$L \
            --gan_optim_lr_g=$GAN_OPTIM_LR_G \
            --gan_optim_lr_d=$GAN_OPTIM_LR_D \
            --gan_optim_beta1=$GAN_OPTIM_BETA1 \
            --gan_optim_beta2=$GAN_OPTIM_BETA2 \
            --gan_optim_wd=$GAN_OPTIM_WD \
            --clf_type=$CLF_TYPE \
            --clf_optim_type=$CLF_OPTIM_TYPE \
            --clf_lr=$CLF_LR \
            --clf_wd=$CLF_WD \
            --n_gm_iter=$N_GM_ITER \
            --clf_reset_iter=$CLF_RESET_ITER \
            --per_class_batch_size=$PER_CLASS_BATCH_SIZE \
            --gm_fake_repeat=$GM_FAKE_REPEAT \
            --Q=$Q \
            --Z=$Z \
            --pretrained_clf_ckpt=$PRETRAINED_CLF_CKPT \
            --C=$C \
            --pretrained_regg_ckpt=$PRETRAINED_REGG_CKPT \
            --R=$R \
            --batch_size=$BATCH_SIZE \
            --n_iter=$N_ITER \
            --n_ckpt=$N_CKPT \
            --seed=$SEED
        fi

	# for each 'number of synthetic features for unseen classes'
    for nu in ${N_SYNTH_US[@]}; do
        export N_SYNTH_U=$nu
        # for each 'number of synthetic features for seen classes'
        for ns in ${N_SYNTH_SS[@]}; do
            export N_SYNTH_S=$ns
            export CLF_EXP_REL_PATH=zsl-100k/$(bash scripts/make_clf_exp_name.sh)
            export CLF_EXP_ROOT=${EXP_DIR}/${CLF_EXP_REL_PATH}
	        # train a classifier
	        if [ $TRAIN_CLF -eq 1 ]; then
	            bash ${CODE_DIR}/scripts/run_classifier.sh
	        fi
        done
    done
done


# combine logs of the experiments
if [ $TRAIN_CLF -eq 1 ]; then
    for nu in ${N_SYNTH_US[@]}; do
        export N_SYNTH_U=$nu

        for ns in ${N_SYNTH_SS[@]}; do
            export N_SYNTH_S=$ns
            export CLF_EXP_REL_PATH=zsl-100k/$(bash scripts/make_clf_exp_name.sh)

            python ${CODE_DIR}/combine_results.py \
                --glob_root=${EXP_ROOT}/$(printf "%03d" $EXP_NO) \
                --log_file=${CLF_EXP_REL_PATH}/avg-scores/logs.npz \
                --save_dir=${EXP_ROOT}/$(printf "%03d/%s" $EXP_NO "${CLF_EXP_REL_PATH}") \
                --keys="gzslh/mean,gzslh/std,gzsls/mean,gzsls/std,gzslu/mean,gzslu/std,zsl/mean,zsl/std"

        done
    done
fi

