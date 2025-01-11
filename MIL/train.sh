#!/usr/bin/env bash

set -x

# frozen
TASK_CONFIG='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/configs/mh_9.yaml'
DATASET_ROOT='/data/wsi/TCTGC50k-features/gigapath-coarse'
LABEL_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc10k/9_labels'
# DATASET_ROOT='/home1/wsi/gc-all-features/frozen/gigapath1'
# LABEL_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc/n-labels'

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
# TITLE="2625_gigapath_oh_5_${LOSS}_${BATCH_SIZE}b_${SAME_PSIZE}PSIZE_${IMBALANCE_SAMPLER}IS_${MIL_METHOD}"
TITLE="10k_gigapath_oh_5_${LOSS}_${BATCH_SIZE}b_${SAME_PSIZE}PSIZE_${IMBALANCE_SAMPLER}IS_${MIL_METHOD}_${NONILM}nonilm"

python main.py  --loss ${LOSS} \
                --task_config ${TASK_CONFIG} \
                --neg_weight ${NEG_WEIGHT} \
                --neg_margin ${NEG_MARGIN} \
                --imbalance_sampler ${IMBALANCE_SAMPLER} \
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