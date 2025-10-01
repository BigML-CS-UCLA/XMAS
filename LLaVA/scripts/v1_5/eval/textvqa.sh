#!/bin/bash

LOG_DIR=logs/eval/textvqa
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${BASE_PATH}/data/textvqa/train_images/ \
    --answers-file ${BASE_PATH}/data/eval/textvqa/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log


python -m llava.eval.eval_textvqa \
    --annotation-file ${BASE_PATH}/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${BASE_PATH}/data/eval/textvqa/answers/${CKPT}.jsonl 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

