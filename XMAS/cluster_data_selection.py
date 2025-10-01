import argparse
import json
import random
import time
import numpy as np
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    p = Path(path)                 
    p.parent.mkdir(parents=True, exist_ok=True) 
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def stability_sampling(
    cluster_vals_path,
    stability_vals_path,
    data_path,
    unique2orig_path,
    save_path,
    percentage,
):
    cluster_vals = load_json(cluster_vals_path)
    stability_vals = load_json(stability_vals_path)
    data = load_json(data_path)
    unique2orig = load_json(unique2orig_path)

    # Assign text-only items to special cluster id 
    sharegpt_cluster_id = max(cluster_vals.values()) + 1
    sharegpt_samples = {item["id"]: sharegpt_cluster_id for item in data if "image" not in item.keys()}
    print(f"{len(sharegpt_samples)} number of samples do not have images")
    cluster_vals.update(sharegpt_samples)

    # Build cluster -> ids
    cluster_dict = {}
    for data_id, cluster_id in cluster_vals.items():
        cluster_dict.setdefault(cluster_id, []).append(data_id)

    clusters, counts = np.unique(np.array(list(cluster_vals.values())), return_counts=True)
    amount = (percentage * len(cluster_vals)) // 100

    # Keep original variable creation (even if unused)
    data_ids = list(cluster_vals.keys())
    I = list(cluster_vals.values())
    _ = (data_ids, I)

    # Sort by cluster size, keep those with count > 2
    sorted_idx = np.argsort(counts)
    sorted_idx = sorted_idx[counts[sorted_idx] > 2]
    n = amount
    sorted_idx = [i.item() for i in sorted_idx]

    sampled_ids = []

    # Sample from the smallest clusters first
    for i in range(len(sorted_idx)):
        n_per_cluster = n // (len(sorted_idx) - i)
        sampling_cluster = clusters[sorted_idx[i]]
        cluster_points = cluster_dict[sampling_cluster]

        if sampling_cluster == sharegpt_cluster_id:
            # random sample from ShareGPT/text-only cluster
            sampled = random.sample(cluster_points, n_per_cluster)
        else:
            # pick lowest-stability examples first
            cluster_stability_vals = sorted(
                [(pid, stability_vals[pid]) for pid in cluster_points],
                key=lambda x: x[1],
            )[:n_per_cluster]
            sampled = [pid for pid, _ in cluster_stability_vals]

        sampled_ids.extend(sampled)
        n -= len(sampled)

    # Map to datapoints
    sampled_idx = [unique2orig[i] for i in sampled_ids]
    sampled_datapoints = [data[int(i)] for i in sampled_idx]

    save_json(sampled_datapoints, save_path)
    print(f"***** Saved {percentage}% of original data at: {save_path}")


def random_sampling(
    data_path,
    save_path,
    percentage,
    seed,
):
    """Pure random sampling from the raw dataset."""
    data = load_json(data_path)
    amount = (percentage * len(data)) // 100

    if seed is not None:
        random.seed(seed)

    sampled_datapoints = random.sample(data, min(amount, len(data)))
    save_json(sampled_datapoints, save_path)
    print(f"***** Saved {percentage}% of original data at: {save_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_vals", required=False, help="Path to cluster_vals JSON.")
    p.add_argument("--stability_vals", required=False, help="Path to stability_vals JSON.")
    p.add_argument("--data", required=True, help="Path to full dataset JSON.")
    p.add_argument("--unique2orig", required=False, help="Path to unique2orig JSON.")
    p.add_argument("--save_path", required=True, help="Output JSON path for sampled datapoints.")
    p.add_argument("--percentage", type=int, default=10, help="Sampling percentage. Default: 10.")
    p.add_argument("--random_sampling", action="store_true", help="Use pure random sampling.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (for random sampling).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.random_sampling:
        print("Sampling Randomly!")
        random_sampling(
            data_path=args.data,
            save_path=args.save_path,
            percentage=args.percentage,
            seed=args.seed,
        )
    else:
        print("Sampling using most stable trajectories!")
        if not args.cluster_vals or not args.stability_vals or not args.unique2orig:
            raise ValueError("stability mode requires --cluster-vals, --stability-vals, and --unique2orig.")
        stability_sampling(
            cluster_vals_path=args.cluster_vals,
            stability_vals_path=args.stability_vals,
            data_path=args.data,
            unique2orig_path=args.unique2orig,
            save_path=args.save_path,
            percentage=args.percentage,
        )


if __name__ == "__main__":
    main()
