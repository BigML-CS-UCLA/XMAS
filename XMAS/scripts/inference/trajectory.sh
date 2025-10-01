#!/bin/bash
if [ $# -ne 11 ]; then
    echo "Usage: $0 <DEVICE> <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <CONV_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi

LOG_DIR=logs/tinyllava_trajectories/
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_PROJECT=tinyllava_trajectories

# Assign the arguments to variables
DEVICE="$1"
DATA_PATH="$2"
IMAGE_PATH="$3"
LLM_VERSION="$4"
VT_VERSION="$5"
VT_VERSION2="$6"
CN_VERSION="$7"
CONV_VERSION="$8"
VERSION="$9"
TRAIN_RECIPE="$10"
MODEL_MAX_LENGTH="${11}"

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"
NUM_PROCESSES=$(echo $DEVICE | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$DEVICE accelerate launch --num_processes=$NUM_PROCESSES --num_machines=1 --mixed_precision=fp16 get_trajectory.py \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen\
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length True \
    --output_dir /home/data/llava_datasets/checkpoints/tinyllava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune \
    --method "partial-svd" \
    --attn_matrix "cross" \
    --attn_layer "full" \
    --k 5 \
    --num_hook_layers 24 \
    --combine_attn_method "sum" \
    --singular_choice "vals" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 200 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --seed 42 \
    --run_name tinyllava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune 2>&1 | tee -a ${LOG_DIR}/${current_time}.log
