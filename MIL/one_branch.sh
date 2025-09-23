#!/usr/bin/env bash
set -x
# export CUDA_VISIBLE_DEVICES=1

# K=4 # 4 8 16 32
# ratio=0.1
# min=20
dataset=gc_10k # gc_10k gc_v15
mil_method=transmil # transmil abmil wsi_vit
patch_drop=1
patch_pad=1
weight=1.
batch_size=32 # 32 1
warmup=200
epoch=200
loss=bce # ce bce
frozen=0
multi_label=1
train_ratio=1.
world_size=4 # 4 1
lr_sche=cycle # cosine cycle
lr=$(echo "0.002 * $world_size * $batch_size" | bc)
# pretrain_model_name='ssl_abmil_b1000_4*128_d10'

python main.py --batch_size ${batch_size} --lr ${lr} --num_epoch ${epoch} \
    --datasets ${dataset} \
    --patch_drop ${patch_drop} \
    --patch_pad ${patch_pad} \
    --mil_method ${mil_method} \
    --loss ${loss} \
    --multi_label ${multi_label} \
    --project "test${dataset}/one-branch-valid" \
    --train_ratio ${train_ratio} \
    --world_size ${world_size} \
    --lr_sche ${lr_sche} \
    --title gigapath-${mil_method}-${world_size}xb${batch_size}-ratio${train_ratio}-${loss}-multi${multi_label}-drop${patch_drop}-pad${patch_pad}-epoch${epoch}-${lr_sche} \
    # --project "${dataset}/one-branch-valid" \

    # --loss ce --lr 0.0002 --weight_decay 0.005 \
    # --train_ratio ${train_ratio} \
    # --frozen ${frozen} --pretrain 1 --pretrain_model_path "../Geometric-Harmonization/output-model/${pretrain_model_name}/ssl_abmil_1000.pth"\
    # --title gigapath-${mil_method}-b${batch_size}-bce-epoch${epoch}-frozen${frozen}-${pretrain_model_name} \
    # --title debug \
