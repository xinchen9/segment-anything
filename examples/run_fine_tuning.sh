#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
python fine_tuning.py \
    --epochs 92 \
    --batch_size 80 \
    --model_dir="/home/chenxin4/data/colon_weights"