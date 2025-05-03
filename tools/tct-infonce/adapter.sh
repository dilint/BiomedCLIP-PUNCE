#!/usr/bin/env bash
set -x

cd PatchEncoder
BACKBONE=biomedclip # biomedclip plip clip resnet50 vitB
DATASET=gc
K=50

DATA_DIR=/home1/wsi/gc-224
TRAIN_LABEL=../datatools/TCTGC2625/labels/train_label.csv
PROJECT_NAME=simclr-infonce
OUTPUT_PATH=output-model
NOT_FROZEN=0
BS=32 # 256
TITLE_NAME=${BACKBONE}_simclr_infonce_random${DATASET}_${K}_${NOT_FROZEN}Nfrozen_224_4*${BS}
TITLE_NAME=${BACKBONE}_simclr_infonce_${DATASET}_all_${NOT_FROZEN}Nfrozen_224_2*${BS}
python -m torch.distributed.launch --master_port=10000 --nproc_per_node=2 simclr.py --ddp \
                                                            --dataset=${DATASET} \
                                                            --backbone=${BACKBONE} \
                                                            --data_dir=${DATA_DIR} \
                                                            --train_label_path=${TRAIN_LABEL} \
                                                            --project=${PROJECT_NAME} \
                                                            --model_path=${OUTPUT_PATH} \
                                                            --title=${TITLE_NAME} \
                                                            --workers=0 \
                                                            --seed=2024 \
                                                            --batch_size=${BS} \
                                                            --epochs=100 \
                                                            --not_frozen=${NOT_FROZEN} \
                                                            --wandb