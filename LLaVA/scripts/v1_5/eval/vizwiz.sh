#!/bin/bash

LOG_DIR=logs/eval/vizwiz
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ${BASE_PATH}/data/eval/vizwiz/test \
    --answers-file ${BASE_PATH}/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${BASE_PATH}/data/eval/vizwiz/llava_test.jsonl \
    --result-file ${BASE_PATH}/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file ${BASE_PATH}/data/eval/vizwiz/answers_upload/${CKPT}.json 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log
