#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python test.py --without_wandb --test_ckpt ./ckpt/paper/best_DELU.pkl
