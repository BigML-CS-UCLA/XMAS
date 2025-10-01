#!/bin/bash

LOG_DIR=logs/eval/llavabench
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets

python -m llava.eval.model_vqa \
    --model-path ${BASE_PATH}/checkpoints/${CKPT} \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${BASE_PATH}/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ${BASE_PATH}/data/eval/llava-bench-in-the-wild/images \
    --answers-file ${BASE_PATH}/data/eval/llava-bench-in-the-wild/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

mkdir -p ${BASE_PATH}/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question ${BASE_PATH}/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context ${BASE_PATH}/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        ${BASE_PATH}/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        ${BASE_PATH}/data/eval/llava-bench-in-the-wild/answers/${CKPT}.jsonl \
    --output \
        ${BASE_PATH}/data/eval/llava-bench-in-the-wild/reviews/${CKPT}.jsonl 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log

python llava/eval/summarize_gpt_review.py -f ${BASE_PATH}/data/eval/llava-bench-in-the-wild/reviews/${CKPT}.jsonl 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}.log