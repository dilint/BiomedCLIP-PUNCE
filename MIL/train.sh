#!/usr/bin/env bash

set -x

# frozen
LABEL_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc-2000/onetask_labels'
TASK_CONFIG='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/configs/oh_5.yaml'

# ablation
LOSS='ranking'
NEG_WEIGHT=${1:-1}
NEG_MARGIN=0
IMBALANCE_SAMPLER=1

# construct
TITLE="2625_oh_5_${LOSS}_${NEG_WEIGHT}a_${NEG_MARGIN}m_${IMBALANCE_SAMPLER}i"

python main.py --label_path ${LABEL_PATH} --loss ${LOSS} --task_config ${TASK_CONFIG} --neg_weight ${NEG_WEIGHT} --neg_margin ${NEG_MARGIN} --imbalance_sampler ${IMBALANCE_SAMPLER} --title ${TITLE} --wandb