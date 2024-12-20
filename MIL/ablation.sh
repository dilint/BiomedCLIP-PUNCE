#!/usr/bin/env bash

ulimit -n 2048
export CUDA_VISIBLE_DEVICES=1
# parallel -j 4 ./train.sh ::: 1.0 2.0 5.0 10.0
parallel -j 2 ./train.sh ::: "ce" "bce" 