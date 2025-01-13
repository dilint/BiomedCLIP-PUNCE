#!/usr/bin/env bash

set -x

# 前两个选项来选择 数据集和提取特征方式
DATASET='TCTGC10k'
FEATURES='gigapath-coarse'
if [ "${DATASET}" = "TCTGC10k" ]; then
    TASK_CONFIG='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/configs/mh_9.yaml'
    LABEL_PATH="/data/wsi/TCTGC10k-labels/9_labels"
elif [ "${DATASET}" = "TCTGC2625" ]; then
    TASK_CONFIG='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/configs/oh_5.yaml'
    LABEL_PATH='/data/wsi/TCTGC2625-labels/n-labels'
fi
DATASET_ROOT="/data/wsi/${DATASET}-features/${FEATURES}"

# ablation
LOSS=${1:-'bce'} # ce, bce, softbce, ranking, aploss, focal
NEG_WEIGHT=1
NEG_MARGIN=0
IMBALANCE_SAMPLER=0
BATCH_SIZE=64
INPUT_DIM=1536
SAME_PSIZE=1000
NONILM=2
MIL_METHOD=abmil # abmil transmil transab
TRAIN_VAL=1
LR=$(echo "0.0002 * ${BATCH_SIZE}" | bc)
# construct
TITLE="${DATASET}_${FEATURES}_oh_5_${LOSS}_${BATCH_SIZE}b_${SAME_PSIZE}PSIZE_${IMBALANCE_SAMPLER}IS_${MIL_METHOD}"
# TITLE="10k_gigapath_oh_5_${LOSS}_${BATCH_SIZE}b_${SAME_PSIZE}PSIZE_${IMBALANCE_SAMPLER}IS_${MIL_METHOD}"

python main.py  --loss ${LOSS} \
                --task_config ${TASK_CONFIG} \
                --neg_weight ${NEG_WEIGHT} \
                --neg_margin ${NEG_MARGIN} \
                --imbalance_sampler ${IMBALANCE_SAMPLER} \
                --project='mtl-test' \
                --title ${TITLE} \
                --batch_size ${BATCH_SIZE} \
                --dataset_root ${DATASET_ROOT} \
                --input_dim ${INPUT_DIM} \
                --same_psize ${SAME_PSIZE} \
                --mil_method ${MIL_METHOD} \
                --label_path ${LABEL_PATH} \
                --nonilm ${NONILM} \
                --train_val ${TRAIN_VAL} \
                # --wandb
                # --eval_only