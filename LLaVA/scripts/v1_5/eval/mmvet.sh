#!/bin/bash

LOG_DIR=logs/eval/mmvet
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets

python -m llava.eval.model_vqa \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ${BASE_PATH}/data/eval/mm-vet/images \
    --answers-file ${BASE_PATH}/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

mkdir -p ${BASE_PATH}/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ${BASE_PATH}/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --dst ${BASE_PATH}/data/eval/mm-vet/results/${CKPT}.json 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

