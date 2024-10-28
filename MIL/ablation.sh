#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4
parallel -j 4 ./train.sh ::: 1.0 2.0 5.0 10.0