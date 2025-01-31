#!/usr/bin/env bash
set -x

cd PatchEncoder
BACKBONE=biomedclip
DATASET=gc
K=50

DATA_DIR=/home1/wsi/gc-filter/filter-images/biomed1-meanmil
TRAIN_LABEL=../datatools/TCTGC2625/labels/train_label.csv
PROJECT_NAME=simclr-infonce
OUTPUT_PATH=output-model
TITLE_NAME=${BACKBONE}_simclr_infonce_filter${DATASET}_${K}_224_4*256
python -m torch.distributed.launch --master_port=10000 --nproc_per_node=4 simclr.py --ddp \
                                                            --dataset=${DATASET} \
                                                            --backbone=${BACKBONE} \
                                                            --data_dir=${DATA_DIR} \
                                                            --train_label_path=${TRAIN_LABEL} \
                                                            --project=${PROJECT_NAME} \
                                                            --model_path=${OUTPUT_PATH} \
                                                            --title=${TITLE_NAME} \
                                                            --workers=2 \
                                                            --seed=2024 \
                                                            --batch_size=256 \
                                                            --epochs=200 \
                                                            --wandb