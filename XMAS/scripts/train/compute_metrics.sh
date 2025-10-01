#!/bin/bash

# Assign the arguments to variables
DATA_PATH="/home/data/llava_datasets/data/llava_instruct_80k.json"
IMAGE_PATH="/home/data/llava_datasets/data/coco/train2017"
# LLM_VERSION="stabilityai/stablelm-2-zephyr-1_6b"
LLM_VERSION="/home/nilay/TinyLLaVA_Factory/checkpoints/tiny-llava-stablelm-2-zephyr-1_6b-clip-vit-large-patch14-336-base-finetune-full_80k-another-one/checkpoint-230"
VT_VERSION="openai/clip-vit-large-patch14-336"
VT_VERSION2=""
CN_VERSION="mlp2x_gelu"
CONV_VERSION="phi"
VERSION="base"
TRAIN_RECIPE="common"
MODEL_MAX_LENGTH="2048"

CUDA_VISISBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 combine_compute_metrics.py \
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
    --output_dir ./checkpoints/tiny-llava-${LLM_VARIANT}-${VERSION}-finetune-full_80k-mir_vals\
    --pretrained False \
    --ds_metric None\
    --method None\
    --k 5\
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
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
    --run_name tiny-llava-${LLM_VARIANT}-${VERSION}-finetune-full_80k-another-one
