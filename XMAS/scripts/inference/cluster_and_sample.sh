#!/bin/bash

METRIC="attn_svd"

if [[ "$METRIC" == "attn_svd" ]]; then
  METRIC_FILES="attn_svd_files"
elif [[ "$METRIC" == "loss" ]]; then
  METRIC_FILES="loss_files"
else
  echo "Unsupported METRIC: $METRIC (use 'attn_svd' or 'loss')" >&2
  exit 1
fi

METRIC_DIR="${PWD}/checkpoints/tiny-llava-stablelm-2-zephyr-1_6b-clip-vit-large-patch14-336-base-llava_mix_665k/${METRIC_FILES}/"
STABILITY_SAVE_DIR_PATH="${PWD}/checkpoints/tiny-llava-stablelm-2-zephyr-1_6b-clip-vit-large-patch14-336-base-llava_mix_665k"
K=5

python kmeans_clustering.py \
    --metric "$METRIC" \
    --metric_dir "$METRIC_DIR" \
    --stability_save_dir_path "$STABILITY_SAVE_DIR_PATH" \
    --k $K

PARENT="$(dirname "$PWD")"
SAVE_ROOT="${CKPT_ROOT:-${PARENT}/LLaVA}"

CLUSTER_VALS="${STABILITY_SAVE_DIR_PATH}/${METRIC}_cluster_idxs.json"    # path to saved per datapoint cluster ids
STABILITY_VALS="${STABILITY_SAVE_DIR_PATH}/${METRIC}_stability_vals_top-${K}_singular_vals.json"   # path to saved per datapoint stability values
DATA_PATH="/path/to/your_dataset.json"         # finetune annotation file path
UNIQUE2ORIG="${PWD}/data/unique2orig_665k.json"    # path to unique id (datapoint) to orig indices in the annotation file (llava mix 665k example)
BUDGET=20    # percentage of selected data 
SAVE_PATH="${SAVE_ROOT}/selected_data/${BUDGET}_xmas_sampled_665k.json"     # path to save the final sampled datapoints (llava mix 665k example)

python cluster_data_selection.py \
    --cluster_vals "$CLUSTER_VALS" \
    --stability_vals "$STABILITY_VALS" \
    --data "$DATA_PATH" \
    --unique2orig "$UNIQUE2ORIG" \
    --save_path "$SAVE_PATH" \
    --percentage $PERCENTAGE