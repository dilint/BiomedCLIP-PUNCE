#!/bin/bash


echo "提取特征方式 ：$1"
echo "数据集 ：$2"
echo "输入维度 ：$3"
echo "高风险权重 ：$4"
GPU_ID=0

cd ../
DATASET_PATH=../extract-features/result-final-$2-features/$1
LABEL_PATH=../datatools/tct-$2/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME="$1-mhim(transmil)-$2-trainval"
TEACHER_INIT=./output-model/mil-methods/$1-transmil-$2-trainval
# TITLE_NAME="$1-mhim(transmil)-$2-trainval-$4"
# TEACHER_INIT=./output-model/mil-methods/$1-transmil-$2-trainval-$4
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH  --label_path=$LABEL_PATH --datasets=$2 --high_weight=$4 --cv_fold=1 --input_dim=$3 --teacher_init=$TEACHER_INIT --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --mrh_sche --title=$TITLE_NAME --mask_ratio=0. --mask_ratio_l=0.8 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --attn_layer=0 --seed=2024 --wandb

CHECKPOINT_PATH="output-model/mil-methods/$1-mhim(transmil)-$2-trainval"
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --ckp_path=$CHECKPOINT_PATH --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$3 --model=pure --baseline=selfattn --seed=2024 