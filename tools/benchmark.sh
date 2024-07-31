#!/usr/bin/env bash

FEATURE_NAME=test
DATASET=gc
NUM_DIM=512
TEST=False

cd mil-methods
python benchmark.py ${FEATURE_NAME} --dataset=${DATASET} --num_dim=${NUM_DIM} --test_mode=${TEST}
