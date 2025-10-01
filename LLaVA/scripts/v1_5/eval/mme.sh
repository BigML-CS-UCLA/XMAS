#!/bin/bash

LOG_DIR=logs/eval/mme
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/MME/llava_mme.jsonl \
    --image-folder ${BASE_PATH}/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ${BASE_PATH}/data/eval/MME/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

cd ${BASE_PATH}/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT} 2>&1 | tee -a /home/dangnth/xmas_llava/${LOG_DIR}/${CKPT}_${current_time}.log

cd eval_tool

python calculation.py --results_dir answers/${CKPT} 2>&1 | tee -a /home/dangnth/xmas_llava/${LOG_DIR}/${CKPT}_${current_time}.log
