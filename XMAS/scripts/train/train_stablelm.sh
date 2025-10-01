#!/bin/bash

FINETUNE_DATA_PATH="/path/to/your_dataset.json" #finetune annotation file path
FINETUNE_IMAGE_PATH="/path/to/your/image_dir" #finetune image dir

LLM_VERSION=stabilityai/stablelm-2-zephyr-1_6b # llm path in huggingface
VT_VERSION=openai/clip-vit-large-patch14-336 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template for stablelm is the same as that for phi
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=base #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm

bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"