#!/usr/bin/env bash
set -x

cd MHIM-MIL 

FEATURE_NAME=biomed1 # biomed1 clip1 plip1 
NUM_DIM=512 # 512 512 512
DATASET=TCTGC2625

SEED=2024 # 2024 2025 2026
MODEL_ADAPTER=''
MODEL_ADAPTER_WEIGHT='/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/output-model/simclr-infonce/biomedclip_simclr_infonce_color_filtergc_50_224_4*256/biomedclip_simclr_infonce_color_filtergc_50_224_4*256_epoch200.pt'
DEBUG_MODE=0
# benchmark 
python benchmark.py ${FEATURE_NAME} --dataset=${DATASET} --num_dim=${NUM_DIM} --seed=${SEED} --cpus 0 1 \
        --model_adapter=${MODEL_ADAPTER} \
        --model_adapter_weight=${MODEL_ADAPTER_WEIGHT} \
        --debug_mode=${DEBUG_MODE} \