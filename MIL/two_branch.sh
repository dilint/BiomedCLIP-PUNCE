#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=0

# K=4 # 4 8 16 32
# ratio=0.1
# min=20

mil_method=abmil # transmil abmil wsi_vit
patch_drop=1
weight=1.
batch_size=1
warmup=50
lr=$(echo "0.002 * $batch_size" | bc)

epoch=200

python mainv2.py --batch_size ${batch_size} --lr ${lr} --num_epoch ${epoch} --datasets gc_10k \
    --patch_drop ${patch_drop} --warmup_epoch ${warmup}\
    --mil_method ${mil_method} \
    --project 'gc_10k/two-branch' \
    --title gigapath-${mil_method}-b${batch_size}-bce-epoch${epoch}-warmup${warmup} \
    # --title debug \
