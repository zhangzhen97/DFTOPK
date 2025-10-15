#!/bin/bash
PYTHON_PATH="/share/ad/zq3/lcron/python3_7/bin/python"
SCRIPT_PATH="deep_components/count_time_fin.py"
export CUDA_VISIBLE_DEVICES=0
nums=(5 10 50 100 500 1000)
for num in "${nums[@]}"
do
    k=$((num/2))
    $PYTHON_PATH $SCRIPT_PATH --num=$num --k=$k --cuda=-1
done
