#!/usr/bin/env bash

cd extract-features

FEAT_DIR='gigapath1'
BASE_MODEL='gigapth'

GPU_NUMBERS=4
WSI_ROOT='/home1/wsi/gc-224'
OUTPUT_PATH='result-final-gc-features'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS extract_features_tct.py --base_model=${BASE_MODEL} --dataset=gc --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --target_patch_size 224 224 --multi_gpu --batch_size=32 --num_workers=64
