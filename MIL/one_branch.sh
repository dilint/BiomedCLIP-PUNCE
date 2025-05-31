#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=1

# K=4 # 4 8 16 32
# ratio=0.1
# min=20

mil_method=abmil # transmil abmil wsi_vit
patch_drop=1
weight=1.
batch_size=2
warmup=200
lr=$(echo "0.002 * $batch_size" | bc)

epoch=200

python main.py --batch_size ${batch_size} --lr ${lr} --num_epoch ${epoch} --datasets gc_10k \
    --patch_drop ${patch_drop} \
    --mil_method ${mil_method} \
    --project 'gc_10k/one-branch' \
    --title gigapath-${mil_method}-b${batch_size}-bce-epoch${epoch} \
    # --title debug \
