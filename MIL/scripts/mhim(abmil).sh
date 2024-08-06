#!/bin/bash


echo "提取特征方式 ：$1"
echo "数据集 ：$2"
echo "输入维度 ：$3"
GPU_ID=0
SERVER=2080

cd ../
DATASET_PATH=../extract-features/result-final-$2-features/$1
LABEL_PATH=../datatools/$2/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods-info
TITLE_NAME="$1-mhim(abmil)-$2-trainval-$SERVER"
TEACHER_INIT=./$OUTPUT_PATH/$PROJECT_NAME/$1-abmil-$2-trainval-$SERVER
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --label_path=$LABEL_PATH --datasets=$2 --cv_fold=1 --input_dim=$3 --train_val --teacher_init=$TEACHER_INIT --title=$TITLE_NAME --baseline=attn --num_workers=0 --cl_alpha=0.1 --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --mrh_sche --init_stu_type=fc --mask_ratio=0.5 --mask_ratio_l=0. --seed=2024 --wandb

CHECKPOINT_PATH="$OUTPUT_PATH/$PROJECT_NAME/$1-mhim(abmil)-$2-trainval-$SERVER"
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --ckp_path=$CHECKPOINT_PATH --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$3 --model=pure --baseline=attn --seed=2024
