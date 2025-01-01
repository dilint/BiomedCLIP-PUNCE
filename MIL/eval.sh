#!/usr/bin/env bash

set -x

# frozen
TASK_CONFIG='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/configs/oh_5.yaml'
# DATASET_ROOT='/home1/wsi/gc-all-features/frozen/gigapath1'
# LABEL_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc/n-labels'
DATASET_ROOT='/data/wsi/TCTGC50k-features/gigapath'
LABEL_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc10k/onetask_5_labels'
MODEL_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/output-model/mtl-524/10k_gigapath_oh_5_bce_16b_1000PSIZE_0IS_abmil_nonilm'

# ablation
LOSS=${1:-'bce'} # ce, bce, softbce, ranking, aploss, focal
BATCH_SIZE=32
INPUT_DIM=1536
SAME_PSIZE=1000
NONILM=2
MIL_METHOD=abmil # abmil transmil transab
TRAIN_VAL=0
LR=$(echo "0.0002 * ${BATCH_SIZE}" | bc)

python main.py  --loss ${LOSS} \
                --task_config ${TASK_CONFIG} \
                --model_path ${MODEL_PATH} \
                --batch_size ${BATCH_SIZE} \
                --dataset_root ${DATASET_ROOT} \
                --input_dim ${INPUT_DIM} \
                --same_psize ${SAME_PSIZE} \
                --mil_method ${MIL_METHOD} \
                --label_path ${LABEL_PATH} \
                --nonilm ${NONILM} \
                --train_val ${TRAIN_VAL} \
                --eval_only