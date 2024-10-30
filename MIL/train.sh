#!/usr/bin/env bash

set -x

# frozen
LABEL_PATH='/data/hjl/projects/BiomedCLIP-PUNCE/datatools/gc-2000/onetask_labels'
LOSS='ranking'
LOSS='softbce'
TASK_CONFIG='/data/hjl/projects/BiomedCLIP-PUNCE/MIL/configs/oh_5.yaml'

# ablation
NEG_WEIGHT=${1:-1.0}
NEG_MARGIN=1
IMBALANCE_SAMPLER=0

# construct
TITLE="oh_5_ranking_${NEG_WEIGHT}a_${NEG_MARGIN}m_${IMBALANCE_SAMPLER}i_fscore"
TITLE="oh_5_softbce"

python main.py --label_path ${LABEL_PATH} --loss ${LOSS} --task_config ${TASK_CONFIG} --neg_weight ${NEG_WEIGHT} --neg_margin ${NEG_MARGIN} --imbalance_sampler ${IMBALANCE_SAMPLER} --title ${TITLE} --wandb