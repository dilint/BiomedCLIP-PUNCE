#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=3

K=4 # 4 8 16 32
ratio=0.1
min=20

patch_drop=1
weight=1.
warmup=25

epoch=50
# python mainv2.py --batch_size 1 --num_epoch $epoch --datasets gc_10k \
#     --patch_drop 1 --kmeans-k $K --kmeans-ratio $ratio --kmeans-min $min \
#     --loss_drop_weight $weight \
#     --title gigapath-abmil-bce-drop1-${epoch}e-pretrain-0508-$K-$ratio-$min-lossWeight${weight} \
#     --pretrain 1

python mainv2.py --batch_size 1 --num_epoch ${epoch} --datasets gc_10k \
    --patch_drop ${patch_drop} --warmup_epoch ${warmup}\
    --title gigapath-abmil-bce-drop${patch_drop}-epoch${epoch}-warmup${warmup} \
    # --pretrain 1