#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=3

# K=4 # 4 8 16 32
# ratio=0.1
# min=20

mil_method=abmil # transmil abmil wsi_vit
weight=1.
batch_size=1
lr=$(echo "0.002 * $batch_size" | bc)

epoch=200

python mainv2.py --batch_size ${batch_size} --lr ${lr} --num_epoch ${epoch} --datasets gc_10k \
    --mil_method ${mil_method} \
    --project 'gc_10k/two-branch' \
    --title debug \

    # --title gigapath-${mil_method}-b${batch_size}-bce-epoch${epoch} \
