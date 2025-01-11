#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=1,2
cd 'PatchEncoder'

GPU_NUMBERS=2
FEAT_DIR='rtdetr' # gigapath1
DATASET='gc2625' # 'ngc', 'ubc', 'gc', 'fnac', 'gc2625'
# WSI_ROOT='/data/wsi/TCTGC50k/TCTGC50k-volume2' 
WSI_ROOT='/home1/wsi/gc' 
OUTPUT_PATH='/data/wsi/TCTGC2625-features'
MODEL_PATH='/home/huangjialong/projects/tctcls-lp/det-ljx/best_x7_20240822.onnx'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS \
            --master_port=10000  extract_features_rtdetr.py \
            --dataset=${DATASET} \
            --output_path=$OUTPUT_PATH \
            --feat_dir=$FEAT_DIR \
            --wsi_root=$WSI_ROOT \
            --model_path=$MODEL_PATH \
            --target_patch_size 1280 1280 \
            --batch_size=3 \
            --multi_gpu \
            --num_workers=6 \
            --confidence_thres 0.3 \
            --device_ids 2 3 \