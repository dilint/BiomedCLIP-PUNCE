#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=1,3
cd 'PatchEncoder'

FEAT_DIR='resnet50-tuneR' # 
BASE_MODEL='resnet50' # 'biomedclip', 'resnet50', 'plip', 'clip',
DATASET='gc2625' # 'ngc', 'ubc', 'gc', 'fnac', 'gc2625'

GPU_NUMBERS=4
# WSI_ROOT='/data/wsi/TCTGC50k/TCTGC50k-volume2' 
WSI_ROOT='/home1/wsi/gc-224' 
OUTPUT_PATH='/home1/wsi/gc-all-features/contrastive'
CKP_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/output-model/simclr-infonce/biomedclip_simclr_infonce_filtergc_50_224_4*256/biomedclip_simclr_infonce_filtergc_50_224_4*256_epoch200.pt'
BACKBONE_WEIGHT_PATH='/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/output-model/simclr-infonce/resnet50_simclr_infonce_randomgc_50_1Nfrozen_224_4*32/resnet50_simclr_infonce_randomgc_50_1Nfrozen_224_4*32_epoch200.pt'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS \
            --master_port=12000 extract_features_FM.py \
            --base_model=${BASE_MODEL} \
            --dataset=${DATASET} \
            --output_path=${OUTPUT_PATH} \
            --feat_dir=${FEAT_DIR} \
            --wsi_root=${WSI_ROOT} \
            --ckp_path=${CKP_PATH} \
            --backbone_weight_path=${BACKBONE_WEIGHT_PATH} \
            --target_patch_size 224 224 \
            --multi_gpu \
            --batch_size=16 \
            --num_workers=8 \
            # --with_adapter \
            # --only_load