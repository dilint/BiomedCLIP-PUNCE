#!/bin/bash


echo "提取特征方式 ：$1"
echo "数据集 ：$2"
echo "输入维度 ：$3"
GPU_ID=0

cd ../
DATASET_PATH=../extract-features/result-final-$2-features/$1
LABEL_PATH=../datatools/tct-$2/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=$1-maxmil-$2-trainval
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=$2 --input_dim=$3 --cv_fold=1 --title=$TITLE_NAME --model=maxmil --seed=2024 --train_val --wandb

CHECKPOINT_PATH=output-model/mil-methods/$1-maxmil-$2-trainval
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --ckp_path=$CHECKPOINT_PATH --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$3 --model=maxmil --seed=2024 
