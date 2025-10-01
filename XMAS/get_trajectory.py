import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json

import torch
import torch.nn as nn
from accelerate import PartialState
from tqdm import tqdm
import transformers
from torch.utils.data import DataLoader, Subset

from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.utils import *
from tinyllava.model.load_model import load_pretrained_model


def attention_scores(k_proj, q_proj):
	# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L104

	# Get dimensions
	q_dim = q_proj.shape[-1]  # Last dimension of q_proj
	k_dim = k_proj.shape[-1]  # Last dimension of k_proj

	# Match dimensions if they're not equal
	if q_dim != k_dim:
		if q_dim > k_dim:
			# Repeat k_proj's last dimension to match q_proj
			repeat_factor = q_dim // k_dim
			if q_dim % k_dim != 0:
				raise ValueError(
					f"q_dim ({q_dim}) must be a multiple of k_dim ({k_dim}) for clean"
					" repeating"
				)
			k_proj = torch.repeat_interleave(k_proj, repeat_factor, dim=2)
		else:
			# Repeat q_proj's last dimension to match k_proj
			repeat_factor = k_dim // q_dim
			if k_dim % q_dim != 0:
				raise ValueError(
					f"k_dim ({k_dim}) must be a multiple of q_dim ({q_dim}) for clean"
					" repeating"
				)
			q_proj = torch.repeat_interleave(q_proj, repeat_factor, dim=2)

	# Use matched dimension for scaling
	d_k = q_proj.shape[-1]  # Updated hidden dimension after potential repeating

	# Calculate attention scores
	scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (d_k**0.5)
	scores = torch.softmax(scores, dim=-1)

	return scores


def attn_svd(data, model, training_arguments):
    """Compute attention SVD metrics with multi-GPU support."""
    distributed_state = PartialState()
    device = distributed_state.device
    local_rank = distributed_state.local_process_index

    # Convert to half precision once before the loop rather than each iteration
    model.to(device).half().eval()

    # Initialize fixed random projection matrix for consistent dimensionality reduction
    # Set seed for reproducibility across all processes
    torch.manual_seed(42)
    projection_matrix = None

    local_attn_svd_vals = {}
    dataset = data["train_dataset"]

    with distributed_state.split_between_processes(
        list(range(len(dataset)))
    ) as subset_indices:
        subset = Subset(dataset, subset_indices)
        train_loader = DataLoader(
            subset,
            batch_size=training_arguments.per_device_eval_batch_size,
            collate_fn=data["data_collator"],
            shuffle=False,
            num_workers=training_arguments.dataloader_num_workers,
            pin_memory=True,
        )

        # Pre-allocate hooks outside the loop
        hooks = []
        proj_cache = {"q_proj": [], "k_proj": []}
        attn_outputs = []

        def hook_fn(name, module, input, output):
            if "q_proj" in name.lower():
                if len(proj_cache["q_proj"]) < training_arguments.num_hook_layers:
                    proj_cache["q_proj"].append(output)
            elif "k_proj" in name.lower():
                if len(proj_cache["k_proj"]) < training_arguments.num_hook_layers:
                    proj_cache["k_proj"].append(output)

            # Process when we have collected all projections
            if (
                training_arguments.attn_layer == "full"
                and len(proj_cache["q_proj"]) == training_arguments.num_hook_layers
                and len(proj_cache["k_proj"]) == training_arguments.num_hook_layers
            ) or (
                training_arguments.attn_layer == "last"
                and len(proj_cache["q_proj"]) == 1
                and len(proj_cache["k_proj"]) == 1
            ):
                # Initialize storage for different combination methods
                if "sum" in training_arguments.combine_attn_method:
                    cm_attn_scores = 0
                elif "concat" in training_arguments.combine_attn_method:
                    attn_scores_list = []
                else:
                    raise ValueError(
                        "Unsupported combine attention method:"
                        f" {training_arguments.combine_attn_method}"
                    )

                # Process each attention head
                for layer_idx, (q_proj, k_proj) in enumerate(
                    zip(proj_cache["q_proj"], proj_cache["k_proj"])
                ):
                    # Get attention scores and apply matrix-specific slicing
                    if training_arguments.attn_matrix == "cross":
                        current_scores = attention_scores(q_proj, k_proj)[
                            :, 612:, 35:611]
                    else:
                        current_scores = attention_scores(
                            q_proj, k_proj)[:, 35:, 35:]

                    # Combine based on method
                    if training_arguments.combine_attn_method == "sum":
                        cm_attn_scores += current_scores
                    elif training_arguments.combine_attn_method == "weighted_sum":
                        if layer_idx % 2 == 0:
                            cm_attn_scores += current_scores
                        else:
                            cm_attn_scores -= current_scores
                    elif training_arguments.combine_attn_method == "concat_text":
                        attn_scores_list.append(current_scores)
                    elif training_arguments.combine_attn_method == "concat_vision":
                        attn_scores_list.append(current_scores)
                    else:
                        raise ValueError(
                            "Unsupported combine attention method:"
                            f" {training_arguments.combine_attn_method}"
                        )

                # Final concatenation for concat methods
                if training_arguments.combine_attn_method == "concat_text":
                    cm_attn_scores = torch.cat(
                        attn_scores_list, dim=1
                    )  # Concatenate along text dimension
                elif training_arguments.combine_attn_method == "concat_vision":
                    cm_attn_scores = torch.cat(
                        attn_scores_list, dim=2
                    )  # Concatenate along vision dimension

                attn_outputs.append(cm_attn_scores)
                # Clear cache more efficiently
                proj_cache["q_proj"].clear()
                proj_cache["k_proj"].clear()

        # Register hooks once before the loop
        for name, module in model.named_modules():
            if "language_model.model" in name.lower() and (
                "q_proj" in name.lower() or "k_proj" in name.lower()
            ):
                if "bias" in name.lower():
                    continue
                elif (
                    training_arguments.attn_layer == "last" and "23" not in name.lower()
                ):
                    continue
                match = re.search(r"\.(\d+)\.", name.lower())
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in training_arguments.layer_list:
                        continue
                else:
                    raise ValueError(f"Incorrect name: {name}")

                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: hook_fn(n, mod, inp, out)
                )
                hooks.append(hook)

        print(f"Create hooks for {len(hooks)//2} layers")

        with torch.inference_mode():
            for _, inputs in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                disable=(local_rank != 0),  # Only show progress on rank 0
                mininterval=60  # Update progress bar every 60 seconds
            ):
                ids = inputs["unique_indices"]

                # Move to device and preprocess in one step
                input_ids = inputs["input_ids"].to(device)
                images = inputs["images"].to(
                    dtype=torch.float16, device=device, non_blocking=True
                )

                # Clear previous run data
                proj_cache["q_proj"].clear()
                proj_cache["k_proj"].clear()
                attn_outputs.clear()

                # Forward pass
                _ = model(
                    input_ids=input_ids,
                    images=images,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )

                # Skip processing if we don't have attention data
                if not attn_outputs:
                    continue

                # Use appropriate SVD method
                if training_arguments.method == "svd-full":
                    # Convert to float32 only for SVD computation
                    with torch.amp.autocast(enabled=False, device_type='cuda'):
                        if training_arguments.singular_choice == "vals":
                            metrics = torch.linalg.svdvals(
                                attn_outputs[0].float())
                            # For singular values, get top-k values if requested
                            if training_arguments.k >= 1:
                                metrics = metrics[:, : training_arguments.k]
                        elif training_arguments.singular_choice == "vecs":
                            # Take right matrix (vision) to have a fixed dimension
                            _, _, v = torch.linalg.svd(attn_outputs[0].float())

                            # Initialize projection matrix on first use
                            if projection_matrix is None:
                                # Last dimension of V
                                original_dim = v.shape[-1]
                                projection_matrix = torch.randn(
                                    original_dim,
                                    training_arguments.proj_dim,
                                    dtype=torch.float32,
                                    device=device,
                                )
                                # Normalize columns for better numerical stability
                                projection_matrix = projection_matrix / torch.norm(
                                    projection_matrix, dim=0, keepdim=True
                                )
                                print(
                                    "Initialized projection matrix:"
                                    f" {original_dim} -> {training_arguments.proj_dim}"
                                )

                            # Get top-k singular vectors if requested
                            if training_arguments.k >= 1:
                                v_topk = v[:, : training_arguments.k, :]
                            else:
                                v_topk = v

                            # Project to fixed dimension of d
                            # V_topk shape: [batch_size, k, original_dim]
                            # projection_matrix shape: [original_dim, d]
                            # Result shape: [batch_size, k, d]
                            metrics = torch.matmul(v_topk, projection_matrix)

                            # Flatten to get final representation
                            metrics = torch.flatten(
                                metrics, start_dim=1, end_dim=2)
                        else:
                            raise ValueError(
                                "Unsupported singular choice:"
                                f" {training_arguments.singular_choice}"
                            )

                    # Check for numerical instability only on rank 0
                    if local_rank == 0:
                        if torch.isnan(metrics).any():
                            print("Found Nans !!!!")
                        if torch.isinf(metrics).any():
                            print("Found Infs !!!!")
                elif training_arguments.method == "partial-svd":
                    with torch.amp.autocast(enabled=False, device_type='cuda'):
                        _, metrics, _ = torch.svd_lowrank(
                            attn_outputs[0].float(), q=training_arguments.k
                        )
                else:
                    raise ValueError(
                        f"Incorrect method: {training_arguments.method}")

                # Store results efficiently
                for unique_id, metric in zip(ids, metrics):
                    local_attn_svd_vals[unique_id] = (
                        metric.cpu().tolist()
                    )  # Move to CPU before converting

                # Explicit cleanup to prevent memory leaks
                torch.cuda.empty_cache()

        # Remove hooks after processing
        for hook in hooks:
            hook.remove()

    return local_attn_svd_vals


def loss(data, model, training_arguments):
    """Compute loss with multi-GPU support."""
    distributed_state = PartialState()
    device = distributed_state.device
    local_rank = distributed_state.local_process_index

    # Convert to half precision once and set eval mode
    model.to(device).eval()

    local_losses = {}
    dataset = data["train_dataset"]

    with distributed_state.split_between_processes(
        list(range(len(dataset)))
    ) as subset_indices:
        subset = Subset(dataset, subset_indices)
        train_loader = DataLoader(
            subset,
            batch_size=training_arguments.per_device_eval_batch_size,
            collate_fn=data["data_collator"],
            shuffle=False,
            num_workers=training_arguments.dataloader_num_workers,
            pin_memory=True,
        )

        # Define constants outside the loop
        ignore_index = -100
        log_interval = 10000

        with torch.inference_mode():
            for idx, inputs in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                disable=(local_rank != 0),  # Only show progress on rank 0
                mininterval=60  # Update progress bar every 60 seconds
            ):
                unique_indices = inputs.pop("unique_indices")

                # Move data to device efficiently with non-blocking transfers
                # Process required tensors only
                images = inputs["images"].to(device, non_blocking=True)
                input_ids = inputs["input_ids"].to(device, non_blocking=True)
                labels = inputs["labels"].to(device, non_blocking=True)
                attention_mask = inputs["attention_mask"].to(
                    device, non_blocking=True)

                # Simplified conditional for position_ids and past_key_values
                position_ids = inputs.get("position_ids")
                past_key_values = inputs.get("past_key_values")

                if position_ids is not None:
                    position_ids = position_ids.to(device, non_blocking=True)
                if past_key_values is not None:
                    past_key_values = past_key_values.to(
                        device, non_blocking=True)

                # Prepare inputs in a single call
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )

                # Reuse the original inputs dictionary to avoid creating a new one
                inputs["input_ids"] = input_ids
                inputs["position_ids"] = position_ids
                inputs["attention_mask"] = attention_mask
                inputs["past_key_values"] = past_key_values
                inputs["inputs_embeds"] = inputs_embeds
                inputs["labels"] = labels

                # Use mixed precision for the forward pass if supported by the hardware
                with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type='cuda'):
                    outputs = model(**inputs)

                # Get necessary variables from outputs only once
                logits = outputs.logits

                # Get dimensions for reshaping operations
                batch_size = labels.size(0)
                seq_length = labels.size(1) - 1
                vocab_size = logits.size(-1)

                # Optimize reshaping operations
                shift_logits = logits[..., :-1, :].reshape(-1, vocab_size)
                shift_labels = labels[..., 1:].reshape(-1)

                # Calculate loss efficiently - only convert types where needed
                token_level_loss = nn.functional.cross_entropy(
                    shift_logits.float(),  # Convert logits to float32 for stability
                    shift_labels,
                    ignore_index=ignore_index,
                    reduction="none",
                )

                # Reshape and compute sequence-level loss
                token_level_loss = token_level_loss.view(
                    batch_size, seq_length)

                # Create mask directly instead of reshaping
                active_loss_mask = shift_labels.view(
                    batch_size, -1) != ignore_index

                # Compute token counts and normalized loss efficiently
                active_tokens_per_sequence = active_loss_mask.sum(dim=1)
                per_sequence_loss = token_level_loss.sum(dim=1)
                normalized_per_sequence_loss = (
                    per_sequence_loss / active_tokens_per_sequence.clamp(min=1)
                )  # Avoid division by zero

                # Store losses directly without creating intermediate dictionary
                for uid, loss_val in zip(unique_indices, normalized_per_sequence_loss):
                    local_losses[uid] = loss_val.item()

                # Log at specified intervals
                if (
                    idx == 1 or (idx != 0 and idx % log_interval == 0)
                ) and local_rank == 0:
                    print(f"***** Predict-Progress -- {idx} DONE !")

                # Clear variables to free memory
                # Use del instead of setting to None for more immediate memory release
                del images, input_ids, labels, attention_mask, inputs_embeds
                del position_ids, past_key_values, inputs, outputs, logits
                del shift_logits, shift_labels, token_level_loss
                del per_sequence_loss, active_loss_mask, active_tokens_per_sequence
                del normalized_per_sequence_loss

                # Only empty cache occasionally to reduce overhead
                if idx % 50 == 0:  # Adjust frequency based on your memory constraints
                    torch.cuda.empty_cache()

    return local_losses


def main(argv):
    sys.argv = argv
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    disable_torch_init()
    model_path = os.path.expanduser(model_args.model_name_or_path)
    model, tokenizer, image_processor, _ = load_pretrained_model(model_path)

    data_args.image_processor = image_processor
    data_args.is_multimodal = True

    data = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        subset_size=(
            training_args.debug_subset_size if training_args.debug_code else None
        ),
    )

    base_model_dir = model_args.model_name_or_path.split("/checkpoint-")[0]
    ckpt_name = model_args.model_name_or_path.split("/checkpoint-")[-1]
    attn_svd_dirname = "attn_svd_files"
    if training_args.attn_matrix == "cross":
        pass
    elif training_args.attn_matrix == "full":
        attn_svd_dirname = "full_" + attn_svd_dirname
    else:
        raise ValueError("Incorrect value of attn_matrix passed!")
    if training_args.attn_layer == "full":
        attn_svd_dirname = f"{model_args.num_hook_layers}layers_" + \
            attn_svd_dirname
        attn_svd_dirname = (
            f"{training_args.combine_attn_method}_" + attn_svd_dirname
        )
    elif training_args.attn_layer == "last":
        attn_svd_dirname = "last_layer_" + attn_svd_dirname
    else:
        raise ValueError("Incorrect value of attn_layer passed!")

    metric_dir = os.path.join(base_model_dir, attn_svd_dirname)
    loss_file_dir = os.path.join(base_model_dir, "loss_files")
    
    if "/" in ckpt_name:
        ckpt_name = ckpt_name.replace("/", "")

    if local_rank in [0, -1]:
        os.makedirs(metric_dir, exist_ok=True)
        os.makedirs(loss_file_dir, exist_ok=True)
        num_devices = torch.cuda.device_count()
        print(f"Computing {training_args.method} using {num_devices} devices!")

    # Get rank-specific filename suffix
    rank_suffix = f"_{local_rank}" if local_rank != -1 else ""

    if training_args.method == "loss":
        loss_file = os.path.join(
            loss_file_dir, "checkpoint-" +
            ckpt_name + f"_loss{rank_suffix}.json"
        )
        all_checkpoints_losses = loss(
            data=data, model=model, training_arguments=training_args
        )

        with open(loss_file, "w") as f:
            json.dump(all_checkpoints_losses, f, indent=4)
        print(f"***** Losses saved to {loss_file}")
    else:
        if training_args.method == "svd-full":
            svd_name = "full"
        elif training_args.method == "partial-svd":
            svd_name = "partial"
        else:
            raise ValueError("Incorrect value of method passed!")
        svd_name += "_" + training_args.singular_choice
        if training_args.k >= 1:
            svd_name += f"_top{training_args.k}"

        metric_file = os.path.join(
            metric_dir,
            "checkpoint-" + ckpt_name + f"_attn_svd_{svd_name}{rank_suffix}.json",
        )
        training_args.num_hook_layers = model_args.num_hook_layers
        training_args.layer_list = [
            int(num) for num in model_args.layer_list.split(",")
        ]
        if training_args.attn_layer == "full":
            assert len(training_args.layer_list) == training_args.num_hook_layers, (
                f"{training_args.layer_list} should have the length of"
                f" {training_args.num_hook_layers}"
            )
        all_checkpoints_metrics = attn_svd(
            data=data, model=model, training_arguments=training_args
        )

        with open(metric_file, "w") as f:
            json.dump(all_checkpoints_metrics, f, indent=4)
        print(f"***** Metrics saved to {metric_file}")


if __name__ == "__main__":
    main(argv=sys.argv)
