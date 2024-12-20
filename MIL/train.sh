#!/usr/bin/env bash

set -x

# frozen
TASK_CONFIG='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/configs/oh_5.yaml'
DATASET_ROOT='/home1/wsi/gc-all-features/frozen/gigapath1'

# ablation
LOSS=${1:-'ce'} # ce, bce, softbce, ranking, aploss
NEG_WEIGHT=1
NEG_MARGIN=0
NONILM=0
IMBALANCE_SAMPLER=0
BATCH_SIZE=16
INPUT_DIM=1536
SAME_PSIZE=1000
MIL_METHOD=transab # abmil transmil transab
LR=$(echo "0.0002 * ${BATCH_SIZE}" | bc)
# construct
# TITLE="2625_oh_5_${LOSS}_${NEG_WEIGHT}a_${NEG_MARGIN}m_${IMBALANCE_SAMPLER}i"
TITLE="2625_gigapath_oh_5_${LOSS}_${BATCH_SIZE}b_${SAME_PSIZE}PSIZE_${NONILM}nonilm_${IMBALANCE_SAMPLER}IS_${MIL_METHOD}"

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
                --nonilm ${NONILM} \
                --wandb