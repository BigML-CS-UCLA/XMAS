#!/bin/bash

LOG_DIR=logs/sft/
mkdir -p $LOG_DIR
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_PROJECT=llava_lora_sft
DATASET=llava_v1_5_mix665k

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --nnodes 1 -m llava.train.train_mem \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/llava_datasets/data/${DATASET}.json \
    --image_folder /data/llava_datasets/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/llava_datasets/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/llava_datasets/checkpoints/llava-v1.5-7b-lora-${DATASET} \
    --run_name llava-v1.5-7b-lora-${DATASET} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --ddp_find_unused_parameters False \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 2>&1 | tee -a ${LOG_DIR}/${current_time}.log

