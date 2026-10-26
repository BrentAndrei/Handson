import os
import json
import numpy as np
import tensorflow as tf

def load_dataset(path, consolidate=False):
    with open(path, "r") as f:
        data = json.load(f)
    feature_dim = data["featureDim"]
    window_size = data["windowSize"]
    if consolidate:
        remap = {}
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            remap[ch] = "FINGERSPELL"
        remap["CONTOL"] = "CONTROL"
        remap["MAYBER"] = "MAYBE"
        for sample in data["samples"]:
            if sample["label"] in remap:
                sample["label"] = remap[sample["label"]]
    labels = sorted(set(s["label"] for s in data["samples"]))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    X = np.zeros((len(data["samples"]), window_size, feature_dim), dtype=np.float32)
    y = np.zeros((len(data["samples"]),), dtype=np.int64)
    for i, sample in enumerate(data["samples"]):
        frames = np.asarray(sample["frames"], dtype=np.float32)
        expected = window_size * feature_dim
        if frames.size != expected:
            raise ValueError(
                f"Sample {i} has {frames.size} floats, expected {expected} "
                f"({window_size} frames x {feature_dim} features).")
        X[i] = frames.reshape(window_size, feature_dim)
        y[i] = label_to_idx[sample["label"]]
    clip_ids = [sample.get("clip_id") for sample in data["samples"]]
    return X, y, labels, feature_dim, window_size, clip_ids

def split_by_clip(y, clip_ids, val_split, seed):
    clips_by_class = {}
    missing_clip_ids = 0
    for sample_idx, (label_idx, clip_id) in enumerate(zip(y, clip_ids)):
        if clip_id is None:
            clip_id = f"legacy_window_{sample_idx}"
            missing_clip_ids += 1
        clips_by_class.setdefault(int(label_idx), {}).setdefault(str(clip_id), []).append(sample_idx)
    train_idx, val_idx = [], []
    rng = np.random.default_rng(seed)
    for clip_groups in clips_by_class.values():
        clip_keys = np.array(list(clip_groups))
        rng.shuffle(clip_keys)
        n_val_clips = max(1, int(len(clip_keys) * val_split)) if len(clip_keys) > 1 else 0
        for clip_id in clip_keys[:n_val_clips]:
            val_idx.extend(clip_groups[clip_id])
        for clip_id in clip_keys[n_val_clips:]:
            train_idx.extend(clip_groups[clip_id])
    return np.array(train_idx), np.array(val_idx), missing_clip_ids

dataset_path = "combined_dataset.json"
X, y, labels, feature_dim, window_size, clip_ids = load_dataset(dataset_path, consolidate=False)
print(f"Loaded {len(X)} windows across {len(labels)} labels")
train_idx, val_idx, missing = split_by_clip(y, clip_ids, val_split=0.25, seed=42)
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
# Take a subset of the validation set for quick evaluation
subset_size = 10
if len(val_idx) > subset_size:
    val_idx_subset = val_idx[:subset_size]
else:
    val_idx_subset = val_idx
X_val_subset = X[val_idx_subset]
y_val_subset = y[val_idx_subset]
print(f"Using subset of {len(X_val_subset)} validation samples for evaluation")
model_path = "model_out/fsl_model.h5"
model = tf.keras.models.load_model(model_path)
loss, acc = model.evaluate(X_val_subset, y_val_subset, verbose=0)
print(f"Validation accuracy (subset): {acc:.4f}")
