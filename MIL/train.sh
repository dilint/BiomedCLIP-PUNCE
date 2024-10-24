#!/usr/bin/env bash

set -x

# frozen
LABEL_PATH='/data/hjl/projects/BiomedCLIP-PUNCE/datatools/gc-2000/onetask_labels'
LOSS='ranking'
TASK_CONFIG='/data/hjl/projects/BiomedCLIP-PUNCE/MIL/configs/oh_5.yaml'

# ablation
NEG_WEIGHT=${1:-1.0}

# construct
TITLE="oh_5_ranking_${NEG_WEIGHT}a_negmargin"

python main.py --label_path ${LABEL_PATH} --loss ${LOSS} --task_config ${TASK_CONFIG} --neg_weight ${NEG_WEIGHT} --title ${TITLE} --wandb