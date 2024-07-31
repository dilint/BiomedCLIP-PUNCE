#!/usr/bin/env bash

GPU_ID=0
SERVER=3080
FEAT_DIR=gigapath1
DATASET=gc
INPUT_DIM=1536

cd mil-methods
DATASET_PATH=../extract-features/result-final-$DATASET-features/$FEAT_DIR
LABEL_PATH=../datatools/$DATASET/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods-info
TITLE_NAME=$FEAT_DIR-meanmil-$DATASET-trainval-$SERVER
# CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=$DATASET --input_dim=$INPUT_DIM --cv_fold=1 --title=$TITLE_NAME --model=meanmil --seed=2024 --train_val --wandb

K=50

FEATURE_ROOT=../extract-features/result-final-gc-features/$FEAT_DIR
WSI_ROOT=/home1/wsi/gc-224
TRAIN_LABEL=../datatools/gc/labels/train_val.csv
CKP_PATH=$OUTPUT_PATH/$PROJECT_NAME/$TITLE_NAME/fold_0_model_best_auc.pt
OUTPUT_ROOT=/home1/wsi/gc-filter-features/$FEAT_DIR-meanmil-$K
# attention! --save_feat depend save images or save features
python inference-multi.py --save_feat --input_dim=$INPUT_DIM --datasets=gc --feature_root=$FEATURE_ROOT --wsi_root=$WSI_ROOT --output_root=$OUTPUT_ROOT --train_label=$TRAIN_LABEL --ckp_path=$CKP_PATH --topk_num=$K  --model=meanmil 