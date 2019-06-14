#!/bin/bash
[[ -z "${FEATURE_NORM}" ]] && FEATURE_NORM='none'
CLF_EXP_NAME=${CLF_TYPE}/${FEATURE_NORM}/${CLF_OPTIM_TYPE}/${CLF_LR}_${CLF_WD}/nU-$(printf "%04d" ${N_SYNTH_U})_nS-$(printf "%04d" ${N_SYNTH_S})
echo ${CLF_EXP_NAME}
