import os
import re
import time
import argparse
from pathlib import Path
from itertools import cycle
import subprocess

import torch


def get_gpus_with_free_memory(min_free_memory_gb):
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free_memory = torch.cuda.mem_get_info()[0]  # Returns (free, total)
        free_memory_gb = free_memory / (1024**3)
        
        if free_memory_gb > min_free_memory_gb:
            available_gpus.append(i)
    return available_gpus


def get_checkpoint_list(model_path):
    """Get list of available checkpoint numbers from the model path"""
    checkpoints = []
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    
    for item in os.listdir(model_path):
        match = checkpoint_pattern.match(item)
        if match and os.path.isdir(os.path.join(model_path, item)):
            checkpoints.append(int(match.group(1)))
    
    return sorted(checkpoints)


def run_distributed_checkpoints(model_path, checkpoint_list, min_memory_gb=40):
    # Get available GPUs
    gpus = get_gpus_with_free_memory(min_memory_gb)
    
    if not gpus:
        raise RuntimeError("No GPUs available!")
    
    print(f"Found {len(gpus)} GPUs: {gpus}")
    
    # Create a cycle of available GPUs
    gpu_cycle = cycle(gpus)
    
    # Track running processes
    running_tasks = {}  # {process: (gpu_id, checkpoint)}
    
    # Process all checkpoints
    while checkpoint_list or running_tasks:
        # Start new processes if GPUs are available and there are checkpoints to process
        # while checkpoint_list and len(running_tasks) < len(gpus):
        while checkpoint_list and len(running_tasks) < len(gpus):
            gpu_id = next(gpu_cycle)
            base_path = os.path.join(model_path, "checkpoint-")
            ckpt = checkpoint_list.pop(0)
            
            cmd = [
                "bash", "scripts/inference/get_trajectory.sh",
                str(gpu_id), 
                ]
            
            cmd.append(base_path + str(ckpt))
            
            process = subprocess.Popen(" ".join(cmd), shell=True)
            
            running_tasks[process] = (gpu_id, ckpt)
        
        # Check for completed processes
        for process in list(running_tasks.keys()):
            if process.poll() is not None:  # Process has finished
                gpu_id, ckpt = running_tasks[process]
                if process.returncode == 0:
                    print(f"Checkpoint {ckpt} completed successfully on GPU {gpu_id}")
                else:
                    print(f"Checkpoint {ckpt} failed on GPU {gpu_id} with return code {process.returncode}")
                del running_tasks[process]
        
        # Small sleep to prevent CPU overload
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Either local model directory (with config) or HuggingFace model path')
    parser.add_argument('--checkpoints', type=str, default='all',
                        help='Comma-separated list of checkpoint numbers, range in format start:end:step, or "all" to process all checkpoints')
    parser.add_argument('--min_memory_gb', type=int, default=40,
                        help='Minimum GPU memory requirement')
    
    args = parser.parse_args()
    
    # Parse checkpoint list
    if args.checkpoints.lower() == 'all':
        checkpoint_list = get_checkpoint_list(args.model_path)
        print(f"Found {len(checkpoint_list)} checkpoints: {checkpoint_list}")
    elif ':' in args.checkpoints:
        start, end, step = map(int, args.checkpoints.split(':'))
        checkpoint_list = list(range(start, end + 1, step))
    else:
        checkpoint_list = [int(x) for x in args.checkpoints.split(',')]

    print("The checkpoint list that is getting passed into the run_dsitributed_trajectories function: ", checkpoint_list)
    run_distributed_checkpoints(args.model_path, checkpoint_list, min_memory_gb=args.min_memory_gb) 
    
# Process all checkpoints found in the model directory:
# python run_distributed_trajectories.py --model_path /path/to/model --checkpoints all

# For a comma-separated list of checkpoints:
# python run_distributed_trajectories.py --model_path /path/to/model --checkpoints 1000,2000,3000,4000

# Or using a range:
# python run_distributed_trajectories.py --model_path /path/to/model --checkpoints 1000:5000:1000