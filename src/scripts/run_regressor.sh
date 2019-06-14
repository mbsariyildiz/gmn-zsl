
python ${CODE_DIR}/train_regressor.py \
	--dataset=$DATASET \
	--data_dir=$DATA_DIR \
	--exp_root=$REGG_EXP_ROOT \
	--mode=$MODE \
	--device='cuda' \
    --lr=$REGG_LR \
    --wd=$REGG_WD \
	--n_epoch=$REGG_N_EPOCH \
	--batch_size=$REGG_BATCH_SIZE
