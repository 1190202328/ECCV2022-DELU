#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_name DELU \
  --seed 0 \
  --alpha_edl 1.3 \
  --alpha_uct_guide 0.4 \
  --amplitude 0.7 \
  --alpha2 0.4 \
  --interval 50 \
  --max_seqlen 320 \
  --lr 0.00005 \
  --k 7 \
  --dataset_name Thumos14reduced \
  --num_class 20 \
  --use_model DELU \
  --max_iter 5000 \
  --dataset SampleDataset \
  --weight_decay 0.001 \
  --AWM BWA_fusion_dropout_feat_v2 \
  --without_wandb \
  --save_dir ./ckpt/DELU_refine