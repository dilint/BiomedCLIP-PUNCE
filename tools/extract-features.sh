#!/usr/bin/env bash

cd 'Patch encoder'

FEAT_DIR='gigapath' # gigapath1
BASE_MODEL='gigapath' # 'biomedclip', 'resnet50', 'resnet34', 'resnet18', 'plip', 'clip', 'dinov2', 'gigapath', 'mae'
DATASET='gc-2000' # 'ngc', 'ubc', 'gc', 'fnac', 'gc-2000'

GPU_NUMBERS=4
WSI_ROOT='/data/wsi/TCTGC50k/TCTGC50k-volume1' 
OUTPUT_PATH='/data/wsi/TCTGC50k-features'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS extract_features_tct.py --base_model=${BASE_MODEL} --dataset=${DATASET} --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --target_patch_size 224 224 --multi_gpu --batch_size=16 --num_workers=32
