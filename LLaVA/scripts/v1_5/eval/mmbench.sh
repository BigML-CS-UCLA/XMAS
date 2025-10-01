#!/bin/bash

LOG_DIR=logs/eval/mmbench
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ${BASE_PATH}/data/eval/mmbench/answers/$SPLIT/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

mkdir -p ${BASE_PATH}/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${BASE_PATH}/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ${BASE_PATH}/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ${BASE_PATH}/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT} 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log
