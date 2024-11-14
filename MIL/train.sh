#!/usr/bin/env bash

set -x

# frozen
LABEL_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc/n-labels'
TASK_CONFIG='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/configs/oh_5.yaml'
DATASET_ROOT='/home1/wsi/gc-all-features/frozen/gigapath1'

# ablation
LOSS=${1:-'bce'}
NEG_WEIGHT=1
NEG_MARGIN=0
IMBALANCE_SAMPLER=1
BATCH_SIZE=32
INPUT_DIM=1536
SAME_PSIZE=1000

# construct
# TITLE="2625_oh_5_${LOSS}_${NEG_WEIGHT}a_${NEG_MARGIN}m_${IMBALANCE_SAMPLER}i"
TITLE="2625_gigapath_oh_5_${LOSS}"

python main.py --label_path ${LABEL_PATH} \
                --loss ${LOSS} \
                --task_config ${TASK_CONFIG} \
                --neg_weight ${NEG_WEIGHT} \
                --neg_margin ${NEG_MARGIN} \
                --imbalance_sampler ${IMBALANCE_SAMPLER} \
                --title ${TITLE} --batch_size=${BATCH_SIZE} \
                --dataset_root=${DATASET_ROOT} \
                --input_dim=${INPUT_DIM} \
                --same_psize=${SAME_PSIZE} \
                --wandb