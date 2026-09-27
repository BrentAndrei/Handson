"""
Diagnostic script to investigate GOOD MORNING/GOOD AFTERNOON/GOOD EVENING
performance in model_out_filtered_v2.

1. Shows per-class top-2 confusion for these 3 labels.
2. Splits validation samples by source (FSL-105 vs Roboflow) using clip_id
   pattern, then reports per-source accuracy for these 3 labels.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

DATASET_PATH = Path(r"C:\Users\BRENT\Downloads\handson_fixed\handson\training\combined_dataset_filtered.json")
MODEL_PATH = Path(r"C:\Users\BRENT\Downloads\handson_fixed\handson\training\model_out_filtered_v2\fsl_model.h5")
LABEL_MAP_PATH = Path(r"C:\Users\BRENT\Downloads\handson_fixed\handson\training\model_out_filtered_v2\label_map.json")

TARGET_LABELS = ["GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING"]


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feature_dim = data["featureDim"]
    window_size = data["windowSize"]
    expected = window_size * feature_dim
    labels = sorted(set(s["label"] for s in data["samples"]))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    X = []
    y = []
    clip_ids = []
    bad = []
    for i, sample in enumerate(data["samples"]):
        frames = np.asarray(sample["frames"], dtype=np.float32)
        if frames.size != expected:
            bad.append((i, frames.size, expected))
            continue
        X.append(frames.reshape(window_size, feature_dim))
        y.append(label_to_idx[sample["label"]])
        clip_ids.append(sample.get("clip_id", ""))

    if bad:
        print(f"WARNING: {len(bad)} samples with bad dimensions")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), clip_ids, labels


def split_by_clip(y, clip_ids, val_split=0.2, seed=42):
    clips_by_class = defaultdict(dict)
    for i, (label_idx, clip_id) in enumerate(zip(y, clip_ids)):
        if not clip_id:
            clip_id = f"legacy_{i}"
        clips_by_class[int(label_idx)].setdefault(clip_id, []).append(i)

    train_idx, val_idx = [], []
    rng = np.random.default_rng(seed)
    for clip_groups in clips_by_class.values():
        keys = np.array(list(clip_groups.keys()))
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_split)) if len(keys) > 1 else 0
        for k in keys[:n_val]:
            val_idx.extend(clip_groups[k])
        for k in keys[n_val:]:
            train_idx.extend(clip_groups[k])
    return np.array(train_idx), np.array(val_idx)


def determine_source(clip_id):
    if clip_id.startswith("train:") or clip_id.startswith("valid:") or clip_id.startswith("test:"):
        return "Roboflow"
    if clip_id:
        return "FSL-105"
    return "unknown"


def main():
    print("Loading dataset and model...")
    X, y, clip_ids, labels = load_dataset(DATASET_PATH)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    target_indices = [label_to_idx[l] for l in TARGET_LABELS]

    train_idx, val_idx = split_by_clip(y, clip_ids)
    X_val = X[val_idx]
    y_val = y[val_idx]
    val_clip_ids = [clip_ids[i] for i in val_idx]

    print(f"Validation set size: {len(y_val)}")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(labels))))

    print("\n=== Per-class top-2 confusion (target labels only) ===")
    for lbl in TARGET_LABELS:
        i = label_to_idx[lbl]
        row = cm[i].copy()
        row[i] = 0
        top = np.argsort(row)[-2:][::-1]
        parts = []
        for j in top:
            if row[j] > 0:
                parts.append(f"{labels[j]} ({row[j]} times)")
        if parts:
            print(f"{lbl} confused with: {', '.join(parts)}")
        else:
            print(f"{lbl} confused with: none")

    # Source analysis
    print("\n=== Source breakdown for target labels ===")
    for lbl in TARGET_LABELS:
        i = label_to_idx[lbl]
        mask = y_val == i
        sources = [determine_source(cid) for cid, m in zip(val_clip_ids, mask) if m]
        correct = sum(1 for p, t, cid in zip(y_pred, y_val, val_clip_ids) if t == i and p == i)
        source_counts = Counter(sources)
        source_correct = defaultdict(int)
        source_total = defaultdict(int)
        for p, t, cid in zip(y_pred, y_val, val_clip_ids):
            if t == i:
                src = determine_source(cid)
                source_total[src] += 1
                if p == i:
                    source_correct[src] += 1

        print(f"\n{lbl} (total val samples: {sum(source_total.values())}):")
        for src in ["FSL-105", "Roboflow", "unknown"]:
            tot = source_total.get(src, 0)
            corr = source_correct.get(src, 0)
            pct = (corr / tot * 100) if tot > 0 else 0
            print(f"  {src}: {tot} samples, {corr} correct ({pct:.1f}%)")


if __name__ == "__main__":
    main()
