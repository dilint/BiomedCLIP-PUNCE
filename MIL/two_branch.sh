#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3

# K=4 # 4 8 16 32
# ratio=0.1
# min=20
dataset=gc_20k # gc_10k gc_v15 gc_2625
mil_method=vit_nc25 # transmil abmil wsi_vit vit_nc25
weight=1.
batch_size=16 # 32 1
warmup_epoch=50
epoch=100
loss=focal # ce bce focal
frozen=0
multi_label=0
world_size=4 # 4 1
consistency_weight=0.01
lr=$(echo "0.00002 * $world_size * $batch_size" | bc)
# pretrain_model_name='ssl_abmil_b1000_4*128_d10'

python main_binary_twobranch.py --batch_size ${batch_size} --lr ${lr} --num_epoch ${epoch} \
    --datasets ${dataset} \
    --mil_method ${mil_method} \
    --loss ${loss} \
    --multi_label ${multi_label} \
    --project "test${dataset}/10.20-twobranch" \
    --world_size ${world_size} \
    --consistency_weight ${consistency_weight} \
    --warmup_epoch ${warmup_epoch} \
    --title gigapath-${mil_method}-${world_size}xb${batch_size}-${loss}-multi${multi_label}-epoch${epoch}a${warmup_epoch}-lr${lr}-cw${consistency_weight}-pi \
    --eval_only

    # --loss ce --lr 0.0002 --weight_decay 0.005 \
    # --train_ratio ${train_ratio} \
    # --frozen ${frozen} --pretrain 1 --pretrain_model_path "../Geometric-Harmonization/output-model/${pretrain_model_name}/ssl_abmil_1000.pth"\
    # --title gigapath-${mil_method}-b${batch_size}-bce-epoch${epoch}-frozen${frozen}-${pretrain_model_name} \
    # --title debug \