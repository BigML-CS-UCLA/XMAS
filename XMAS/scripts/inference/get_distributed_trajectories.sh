#!/bin/bash

MODEL_PATH="${PWD}/checkpoints/tiny-llava-stablelm-2-zephyr-1_6b-clip-vit-large-patch14-336-base-llava_mix_665k/"
CHECKPOINTS="all"

# One ckpt one GPU
python run_distributed_trajectories.py --model_path "$MODEL_PATH" --checkpoints "$CHECKPOINTS"