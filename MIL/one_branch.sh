#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=1

# K=4 # 4 8 16 32
# ratio=0.1
# min=20
dataset=gc_v15 # gc_10k gc_v15
mil_method=abmil # transmil abmil wsi_vit
patch_drop=1
patch_pad=1
weight=1.
batch_size=1
warmup=200
lr=$(echo "0.002 * $batch_size" | bc)
epoch=200
loss=ce # ce bce
frozen=0
# pretrain_model_name='ssl_abmil_b1000_4*128_d10'

python main.py --batch_size ${batch_size} --lr ${lr} --num_epoch ${epoch} \
    --datasets ${dataset} \
    --patch_drop ${patch_drop} \
    --mil_method ${mil_method} \
    --loss ${loss} \
    --loss ce --lr 0.0002 --weight_decay 0.005 \
    # --project "${dataset}/one-branch-valid" \
    --project "test/one-branch-valid" \
    --title gigapath-${mil_method}-b${batch_size}-${loss}-drop${patch_drop}-pad${patch_pad}-epoch${epoch} \

    # --frozen ${frozen} --pretrain 1 --pretrain_model_path "../Geometric-Harmonization/output-model/${pretrain_model_name}/ssl_abmil_1000.pth"\
    # --title gigapath-${mil_method}-b${batch_size}-bce-epoch${epoch}-frozen${frozen}-${pretrain_model_name} \
    # --title debug \
