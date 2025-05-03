#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=0

K=4 # 4 8 16 32
ratio=0.2
min=20

python mainv2.py --batch_size 1 --num_epoch 200 --datasets gc_10k \
    --patch_drop 1 --kmeans-k $K --kmeans-ratio $ratio --kmeans-min $min \
    --title gigapath-abmil-bce-drop1-200e-0503-$K-$ratio-$min