#!/bin/bash

# Get GPUs with > 40GB free memory (40960 MB)
available_gpus=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$2 > 40960 {print $1}'))

# If CUDA_VISIBLE_DEVICES is set, filter it by available GPUs, otherwise use all available GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Convert CUDA_VISIBLE_DEVICES to array
    IFS=',' read -ra cuda_gpus <<< "$CUDA_VISIBLE_DEVICES"
    
    # Find intersection of CUDA_VISIBLE_DEVICES and GPUs with >40GB free
    filtered_gpus=()
    for gpu in "${cuda_gpus[@]}"; do
        if [[ " ${available_gpus[*]} " =~ " ${gpu} " ]]; then
            filtered_gpus+=("$gpu")
        fi
    done
    
    # Convert back to comma-separated string
    gpu_list=$(IFS=','; echo "${filtered_gpus[*]}")
    IFS=',' read -ra GPULIST <<< "$gpu_list"
else
    # Use all GPUs with >40GB free memory
    gpu_list=$(IFS=','; echo "${available_gpus[*]}")
    IFS=',' read -ra GPULIST <<< "$gpu_list"
fi

CHUNKS=${#GPULIST[@]}

LOG_DIR=logs/eval/gqa
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT=llava-v1.5-7b-lora-llava_v1_5_mix665k
BASE_PATH=/home/data/llava_datasets
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${BASE_PATH}/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${BASE_PATH}/checkpoints/${CKPT} \
        --model-base lmsys/vicuna-7b-v1.5 \
        --question-file ${BASE_PATH}/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ${BASE_PATH}/data/eval/gqa/data/images \
        --answers-file ${BASE_PATH}/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 2>&1 | tee -a ${LOG_DIR}/${CKPT}_${current_time}_${IDX}_of_${CHUNKS}.log &
done

wait

output_file=${BASE_PATH}/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${BASE_PATH}/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced 2>&1 | tee -a /home/dangnth/xmas_llava/${LOG_DIR}/${CKPT}_${current_time}.log
