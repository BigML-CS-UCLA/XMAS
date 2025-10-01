#!/bin/bash

LOG_DIR=logs/eval/pope
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ${BASE_PATH}/data/val2014 \
    --answers-file ${BASE_PATH}/data/eval/pope/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

python llava/eval/eval_pope.py \
    --annotation-dir ${BASE_PATH}/data/eval/pope/coco \
    --question-file ${BASE_PATH}/data/eval/pope/llava_pope_test.jsonl \
    --result-file ${BASE_PATH}/data/eval/pope/answers/${CKPT}.jsonl 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log
