#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd 'PatchEncoder'

FEAT_DIR='plip-coarse' # gigapath1
BASE_MODEL='plip' # 'biomedclip', 'resnet50', 'resnet34', 'resnet18', 'plip', 'clip', 'dinov2', 'gigapath', 'mae'
DATASET='gc' # 'ngc', 'ubc', 'gc', 'fnac', 'gc2625'

GPU_NUMBERS=2
# WSI_ROOT='/data/wsi/TCTGC50k/TCTGC50k-volume2' 
WSI_ROOT='/data/wsi/TCTGC50k/TCTGC50k-volume1' 
OUTPUT_PATH='/data/wsi/TCTGC10k-features'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS \
            --master_port=12000 extract_features_FM.py \
            --base_model=${BASE_MODEL} \
            --dataset=${DATASET} \
            --output_path=$OUTPUT_PATH \
            --feat_dir=$FEAT_DIR \
            --wsi_root=$WSI_ROOT \
            --ckp_path=$CKP_PATH \
            --target_patch_size 224 224 \
            --multi_gpu \
            --batch_size=3 \
            --num_workers=6 \
            # --only_load