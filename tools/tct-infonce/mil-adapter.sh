#!/usr/bin/env bash
set -x

cd MHIM-MIL 

FEATURE_NAME=resnet50-tuneR # biomed1 clip1 plip1 biomed-apapter biomed-ori
NUM_DIM=1024 # 512 512 512
DATASET=TCTGC2625

SEED=2024 # 2024 2025 2026
MODEL_ADAPTER='' # linear
MODEL_ADAPTER_WEIGHT='/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/output-model/simclr-infonce/biomedclip_simclr_infonce_color_filtergc_50_224_4*256/biomedclip_simclr_infonce_color_filtergc_50_224_4*256_epoch200.pt'
# MODEL_ADAPTER_WEIGHT='/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/output-model/simclr-infonce/biomedclip_simclr_infonce_filterGC_50_224_4*256_200/biomedclip_simclr_infonce_filterGC_50_224_4*256_200_epoch200.pt'
FEATURE_ROOT='contrastive' # 'frozen' 'contrastive'
DEBUG_MODE=0
# benchmark 
python benchmark.py ${FEATURE_NAME} --dataset=${DATASET} --num_dim=${NUM_DIM} --seed=${SEED} --cpus 2 3 \
        --model_adapter=${MODEL_ADAPTER} \
        --model_adapter_weight=${MODEL_ADAPTER_WEIGHT} \
        --feature_root=${FEATURE_ROOT} \
        --debug_mode=${DEBUG_MODE} \