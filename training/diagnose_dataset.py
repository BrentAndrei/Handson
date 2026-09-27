"""
Diagnoses two likely causes of the train/val generalization gap seen in
training: (1) near-duplicate/overlapping windows leaking across the
train/val split, and (2) degenerate or NaN-heavy features. Run this in
the training folder with the SAME venv used for training (.venv-train),
since it needs numpy.

Usage:
    python diagnose_dataset.py --dataset fsl_dataset.json
"""

import argparse
import json
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    with open(args.dataset, "r") as f:
        data = json.load(f)

    feature_dim = data["featureDim"]
    window_size = data["windowSize"]
    labels = sorted(set(s["label"] for s in data["samples"]))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    n = len(data["samples"])
    X = np.zeros((n, window_size, feature_dim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    for i, sample in enumerate(data["samples"]):
        frames = np.asarray(sample["frames"], dtype=np.float32)
        X[i] = frames.reshape(window_size, feature_dim)
        y[i] = label_to_idx[sample["label"]]

    print(f"\n=== BASIC STATS ===")
    print(f"Total windows: {n}, labels: {len(labels)}, feature_dim: {feature_dim}, window_size: {window_size}")

    # --- Feature quality checks ---
    print(f"\n=== FEATURE QUALITY ===")
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    print(f"NaN values: {n_nan} ({100*n_nan/X.size:.4f}% of all values)")
    print(f"Inf values: {n_inf} ({100*n_inf/X.size:.4f}% of all values)")

    finite_mask = np.isfinite(X)
    if finite_mask.all():
        flat = X.reshape(-1, feature_dim)
    else:
        flat = X[finite_mask.all(axis=(1, 2))].reshape(-1, feature_dim)

    feat_std = flat.std(axis=0)
    n_dead = (feat_std < 1e-6).sum()
    print(f"Feature dims with near-zero variance (dead features): {n_dead} / {feature_dim}")
    print(f"Global value range: min={flat.min():.4f}, max={flat.max():.4f}, mean={flat.mean():.4f}, std={flat.std():.4f}")

    # --- Breakdown by landmark group, assuming the standard layout:
    # pose(33) + leftHand(21) + rightHand(21) + lips(40), x/y/z each.
    print(f"\n=== DEAD-DIM BREAKDOWN BY LANDMARK GROUP ===")
    groups = [
        ("pose", 0, 33),
        ("leftHand", 33, 21),
        ("rightHand", 33 + 21, 21),
        ("lips", 33 + 21 + 21, 40),
    ]
    for name, point_offset, point_count in groups:
        dim_offset = point_offset * 3
        dim_count = point_count * 3
        group_std = feat_std[dim_offset:dim_offset + dim_count]
        group_dead = (group_std < 1e-6).sum()
        group_mean_abs = np.abs(flat[:, dim_offset:dim_offset + dim_count]).mean()
        print(f"  {name:10s}: {group_dead:3d}/{dim_count} dead dims, "
              f"mean|value|={group_mean_abs:.4f}, mean std={group_std.mean():.5f}, max std={group_std.max():.5f}")

        # Which specific points (not just dims) are fully dead (all 3 of x,y,z dead)?
        dead_points = []
        for p in range(point_count):
            xs = feat_std[dim_offset + p*3: dim_offset + p*3 + 3]
            if (xs < 1e-6).all():
                dead_points.append(p)
        if dead_points:
            print(f"              fully-dead point indices (0-based within group): {dead_points}")

    # windows containing at least one dropped-detection style large outlier
    extreme = (np.abs(X) > 10).any(axis=(1, 2)).sum()
    print(f"Windows with any |value| > 10 (possible unnormalized/garbage frames): {extreme} ({100*extreme/n:.2f}%)")

    # --- Reproduce the exact same stratified split train.py uses ---
    print(f"\n=== SPLIT LEAKAGE CHECK (reproducing train.py's split logic) ===")
    idx_by_class = {}
    for i, label_idx in enumerate(y):
        idx_by_class.setdefault(label_idx, []).append(i)

    train_idx, val_idx = [], []
    rng = np.random.default_rng(args.seed)
    for label_idx, idxs in idx_by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * args.val_split)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    print(f"Train windows: {len(train_idx)}, Val windows: {len(val_idx)}")

    # Near-duplicate check: for each val window, find its nearest neighbor
    # (by L2 distance) in the train set, per class, on a subsample for speed.
    print("\nChecking for near-duplicate windows between train and val")
    print("(this samples up to 30 val windows per class and finds nearest train neighbor)...")

    near_dup_threshold = 0.5  # L2 distance in normalized-ish landmark space; tune if needed
    total_checked = 0
    total_near_dup = 0
    per_class_dup_rate = {}

    for label_idx, idxs in idx_by_class.items():
        val_in_class = [i for i in val_idx if y[i] == label_idx]
        train_in_class = [i for i in train_idx if y[i] == label_idx]
        if not val_in_class or not train_in_class:
            continue
        sample_val = val_in_class[:30]
        train_flat = X[train_in_class].reshape(len(train_in_class), -1)

        n_dup_this_class = 0
        for vi in sample_val:
            v_flat = X[vi].reshape(1, -1)
            dists = np.linalg.norm(train_flat - v_flat, axis=1)
            min_dist = dists.min()
            total_checked += 1
            if min_dist < near_dup_threshold:
                total_near_dup += 1
                n_dup_this_class += 1
        per_class_dup_rate[labels[label_idx]] = n_dup_this_class / len(sample_val)

    print(f"\nChecked {total_checked} val windows across all classes.")
    print(f"Near-duplicates found (L2 dist < {near_dup_threshold} to some train window): "
          f"{total_near_dup} ({100*total_near_dup/max(total_checked,1):.1f}%)")

    if total_near_dup / max(total_checked, 1) > 0.3:
        print(">>> HIGH near-duplicate rate: val set likely contains windows that are")
        print(">>> near-copies of train windows (e.g. overlapping stride windows from")
        print(">>> the same recording take). This would explain train accuracy rising")
        print(">>> while val accuracy stays flat/noisy -- the val set isn't testing")
        print(">>> generalization to genuinely new performances of each sign.")
    else:
        print(">>> Near-duplicate rate looks low -- split leakage is probably NOT the")
        print(">>> main cause of the generalization gap. Look at feature quality above,")
        print(">>> or the gap may just be a genuinely hard 105-class problem needing")
        print(">>> more data/regularization/architecture changes.")

    # Show worst offenders
    sorted_rates = sorted(per_class_dup_rate.items(), key=lambda x: -x[1])
    print("\nTop 10 classes by near-duplicate rate:")
    for label, rate in sorted_rates[:10]:
        print(f"  {label}: {100*rate:.0f}%")


if __name__ == "__main__":
    main()
