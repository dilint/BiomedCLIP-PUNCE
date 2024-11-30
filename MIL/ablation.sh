#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
# parallel -j 4 ./train.sh ::: 1.0 2.0 5.0 10.0
parallel -j 2 ./train.sh ::: "ranking" "aploss"