#!/bin/bash

GPU_ID=3
SERVER=3080

FEATURE_NAME=gigapath-longnet
DATASET=gc
NUM_DIM=768

cd ../
DATASET_PATH=/home1/wsi/gc-all-features/frozen/$FEATURE_NAME
LABEL_PATH=../datatools/$DATASET/n-labels
OUTPUT_PATH=output-model
PROJECT_NAME=test
TITLE_NAME=$FEATURE_NAME-linear-$DATASET-trainval-$SERVER
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --n_classes=5 --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=$DATASET --input_dim=$NUM_DIM --cv_fold=1 --title=$TITLE_NAME --model=linear  --train_val --seed=2024 --wandb

# CHECKPOINT_PATH=$OUTPUT_PATH/$PROJECT_NAME/$FEATURE_NAME-abmil-$DATASET-trainval-$SERVER
# CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$NUM_DIM --model=pure --baseline=attn --seed=2024
