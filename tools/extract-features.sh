#!/usr/bin/env bash

cd 'Patch encoder'

FEAT_DIR='mae1' # gigapath1
BASE_MODEL='mae' # 'biomedclip', 'resnet50', 'resnet34', 'resnet18', 'plip', 'clip', 'dinov2', 'gigapath', 'mae'

GPU_NUMBERS=4
WSI_ROOT='/home1/wsi/gc-224'
OUTPUT_PATH='/home1/wsi/gc-all-features/frozen'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS extract_features_tct.py --base_model=${BASE_MODEL} --dataset=gc --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --target_patch_size 224 224 --multi_gpu --batch_size=32 --num_workers=64
