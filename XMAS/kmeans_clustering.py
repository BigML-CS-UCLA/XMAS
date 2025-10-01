import argparse
import torch
import json
import glob
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

_CKPT_NUM_RE = re.compile(r"checkpoint-(\d+)")


def main(args):
    """
    Load per-checkpoint metric JSONs (loss or attn_svd), aggregate into a
    (num_datapoints, num_checkpoints) tensor, compute simple stability stats,
    and run FAISS KMeans clustering.
    """
    cnt = 0
    metrics = []
    global_ids = []

    metric_dir = Path(args.metric_dir).resolve()
    if not metric_dir.is_dir():
        raise FileNotFoundError(f"--metric_dir not found or not a directory: {metric_dir}")

    if args.metric == "attn_svd":
        pattern = "*_attn_svd_*.json"
    elif args.metric == "loss":
        pattern = "*loss*.json"
    else:
        raise ValueError(f"Incorrect value for --metric: {args.metric}")

    metric_checkpoints = sorted(
        (str(p) for p in metric_dir.glob(pattern)),
        key=lambda p: int(_CKPT_NUM_RE.search(os.path.basename(p)).group(1)) if _CKPT_NUM_RE.search(os.path.basename(p)) else -1,
    )
    print("this si the value fo the detected metric files: ", metric_checkpoints)

    if not metric_checkpoints:
        raise RuntimeError(f"No metric files matching '{pattern}' found in {metric_dir}")

    for ckpt in metric_checkpoints:
        print(f"*** {ckpt} ** Loading metrics...")
        try:
            with open(ckpt, "r") as f:
                data = json.load(f)

            # Establish datapoint order only once, so stacking aligns across checkpoints
            if cnt == 0:
                global_ids = list(data.keys())
                cnt += 1

            # Build a trajectory: vector of metrics per checkpoint 
            if args.metric == "attn_svd":
                k = args.k  
                if k is not None:
                    if k == 1:
                        vals = [sorted(data[_id], reverse=True)[:k][0] for _id in global_ids]
                    else:
                        vals = [sum(sorted(data[_id], reverse=True)[:k]) for _id in global_ids]
                else:
                    vals = [float(np.sum(data[_id])) for _id in global_ids]

                metrics.append(torch.tensor(vals, dtype=torch.float32))

            elif args.metric == "loss":
                vals = [float(data[_id]) for _id in global_ids]
                metrics.append(torch.tensor(vals, dtype=torch.float32))

            else:
                raise ValueError(f"Incorrect value for --metric: {args.metric}")

        except Exception as e:
            print(f"*** {ckpt} ** Could not load metrics. Reason: {e}")
            continue

    if not metrics:
        raise RuntimeError("No metrics were loaded successfully; aborting.")

    # (num_checkpoints, num_datapoints) -> (num_datapoints, num_checkpoints)
    metrics = torch.stack(metrics).t()

    attn_svd_stability_vals = attn_svd_stability(metrics)
    attn_svd_stability_dict = {dp_id: var for dp_id, var in zip(global_ids, attn_svd_stability_vals)}

    if args.stability_save_dir_path:
        out_dir = Path(args.stability_save_dir_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.metric}_stability_vals_top-{args.k}_singular_vals.json"
        with open(out_path, "w") as f:
            json.dump(attn_svd_stability_dict, f, indent=2)
        print(f"***** Saved stability values to: {out_path}")

    metrics[torch.isnan(metrics)] = 0.0

    faiss_kmeans_selection(metrics, global_ids, args)


def faiss_kmeans_selection(features, ids, args):
    """
    Cluster datapoints using FAISS KMeans on the (num_datapoints, num_checkpoints) feature matrix.
    """
    import faiss  

    num_datapoints, num_dims = features.shape
    num_clusters = 1000  
    niter = 100

    print(f"FAISS KMeans: datapoints={num_datapoints}, dims={num_dims}, k={num_clusters}, niter={niter}")
    kmeans = faiss.Kmeans(d=num_dims, k=num_clusters, niter=niter, verbose=True)
    kmeans.train(features.numpy())

    _, I = kmeans.index.search(features.numpy(), 1)  # I: (N, 1) cluster idx per point

    save_root = Path(args.stability_save_dir_path).resolve() 
    save_root.mkdir(parents=True, exist_ok=True)
    cluster_save_path = save_root / f"{args.metric}_cluster_idxs.json"

    cluster_data = {ids[i]: int(I[i, 0]) for i in range(len(ids))}
    with open(cluster_save_path, "w") as f:
        json.dump(cluster_data, f, indent=4)

    print(f"***** Saved cluster assignments to: {cluster_save_path}")


def attn_svd_stability(metrics):
    """
    Per-datapoint trajectory stability (sum of absolute diffs).
    """
    variance_vals = []
    for i in range(len(metrics)):
        traj = metrics[i].numpy()
        variance_vals.append(float(np.sum(np.abs(np.diff(traj)))))
    return variance_vals


def compute_ema(data, alpha=0.1):
    """
    Per-datapoint exponential moving average (EMA) of trajectory.
    """
    if not data:
        return []
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
    return ema


def autocorrelation(signal, normalize=True):
    """
    Per-datapoint trajectory autocorrelation R(τ).
    If normalize=True, scales so that R(0) = 1.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    full_ac = np.correlate(signal, signal, mode="full")
    ac = full_ac[full_ac.size // 2 :]
    if normalize and ac[0] != 0:
        ac = ac / ac[0]
    return ac


def get_variance(metrics):
    """
    Per-datapoint trajectory variance.
    """
    variance_vals = []
    for i in range(len(metrics)):
        traj = metrics[i].numpy()
        variance_vals.append(float(np.var(traj)))
    return variance_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True, choices=["attn_svd", "loss"])
    parser.add_argument("--metric_dir", type=str, required=True, help="Directory containing per-checkpoint metric JSONs")
    parser.add_argument("--stability_save_dir_path", type=str, required=True, help="Optional directory to save stability/cluster outputs")
    parser.add_argument("--k", type=int, default=5, help="Top-k singular values to use for attn_svd; if None, uses sum over trajectory")
    args = parser.parse_args()

    main(args)
