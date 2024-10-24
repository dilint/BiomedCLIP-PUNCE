#!/usr/bin/env bash


parallel -j 4 ./train.sh ::: 1.0 2.0 5.0 10.0