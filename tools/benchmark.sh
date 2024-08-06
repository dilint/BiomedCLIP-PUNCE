#!/usr/bin/env bash

FEATURE_NAME=gigapath1
NUM_DIM=1536
DATASET=gc

SEED=2024 # 2024 2025 2026
cd MIL 
# benchmark 
python benchmark.py ${FEATURE_NAME} --dataset=${DATASET} --num_dim=${NUM_DIM} --seed=${SEED} --cpus 2 3 #--test_mode