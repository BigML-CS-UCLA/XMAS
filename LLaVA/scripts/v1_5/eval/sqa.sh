#!/bin/bash

LOG_DIR=logs/eval/sqa
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets

python -m llava.eval.model_vqa_science \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${BASE_PATH}/data/eval/scienceqa/test \
    --answers-file ${BASE_PATH}/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

python llava/eval/eval_science_qa.py \
    --base-dir ${BASE_PATH}/data/eval/scienceqa \
    --result-file ${BASE_PATH}/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --output-file ${BASE_PATH}/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ${BASE_PATH}/data/eval/scienceqa/answers/${CKPT}.json 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log
