#!/usr/bin/env python3
"""
Merge Kaggle FSL (28) + Transactional NPY (26) + Transactional Video (26) datasets.
Total: 54 classes (no overlap between Transactional NPY and Video - same 26 classes).
"""

import os
import json
import numpy as np

# Load Kaggle data
kaggle_X = np.load("training/data/processed/X.npy")
kaggle_y = np.load("training/data/processed/y.npy")
with open("training/data/processed/label_map.json") as f:
    kaggle_label_map = json.load(f)

# Load Transactional NPY data
trans_npy_X = np.load("training/data/processed_transactional/X.npy")
trans_npy_y = np.load("training/data/processed_transactional/y.npy")
with open("training/data/processed_transactional/label_map.json") as f:
    trans_npy_label_map = json.load(f)

# Load Transactional Video data
trans_vid_X = np.load("training/data/processed_transactional_video/X.npy")
trans_vid_y = np.load("training/data/processed_transactional_video/y.npy")
with open("training/data/processed_transactional_video/label_map.json") as f:
    trans_vid_label_map = json.load(f)

print(f"Kaggle: {len(kaggle_X)} windows, {len(set(kaggle_y))} classes")
print(f"Transactional NPY: {len(trans_npy_X)} windows, {len(set(trans_npy_y))} classes")
print(f"Transactional Video: {len(trans_vid_X)} windows, {len(set(trans_vid_y))} classes")

# Check labels
kaggle_labels = set([kaggle_label_map[str(i)] for i in range(len(kaggle_label_map))])
trans_npy_labels = set([trans_npy_label_map[str(i)] for i in range(len(trans_npy_label_map))])
trans_vid_labels = set([trans_vid_label_map[str(i)] for i in range(len(trans_vid_label_map))])

print(f"Kaggle labels: {sorted(kaggle_labels)}")
print(f"Trans NPY labels: {sorted(trans_npy_labels)}")
print(f"Trans Video labels: {sorted(trans_vid_labels)}")
print(f"NPY intersect Video overlap: {trans_npy_labels & trans_vid_labels}")

# Transactional NPY and Video have the SAME 26 classes, so we should combine them
# Kaggle has 28 different classes
# Total unique: 28 + 26 = 54 classes

# Merge Transactional NPY + Video (same 26 classes)
trans_combined_X = np.concatenate([trans_npy_X, trans_vid_X], axis=0)
trans_combined_y = np.concatenate([trans_npy_y, trans_vid_y], axis=0)

print(f"\nTransactional combined: {len(trans_combined_X)} windows")

# Now combine all: Kaggle (28) + Transactional combined (26)
all_labels = sorted(list(set(list(kaggle_labels) + list(trans_npy_labels))))
label_to_idx = {label: i for i, label in enumerate(all_labels)}

print(f"\nTotal unique labels: {len(all_labels)}")
print(f"Labels: {all_labels}")

# Remap all to unified indices
kaggle_y_new = np.array([label_to_idx[kaggle_label_map[str(l)]] for l in kaggle_y], dtype=np.int32)
trans_combined_y_new = np.array([label_to_idx[trans_npy_label_map[str(l)]] for l in trans_combined_y], dtype=np.int32)

# Combine all
X_combined = np.concatenate([kaggle_X, trans_combined_X], axis=0)
y_combined = np.concatenate([kaggle_y_new, trans_combined_y_new], axis=0)

print(f"\nFinal combined: {len(X_combined)} windows, {len(set(y_combined))} classes")
print(f"Shape: {X_combined.shape}")

# Save combined dataset
os.makedirs("training/data/processed_all", exist_ok=True)
np.save("training/data/processed_all/X.npy", X_combined)
np.save("training/data/processed_all/y.npy", y_combined)

label_map = {str(i): label for i, label in enumerate(all_labels)}
with open("training/data/processed_all/label_map.json", 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"\nSaved to training/data/processed_all/")
print(f"Total classes: {len(all_labels)}")
print(f"Total samples: {len(X_combined)}")

# Class distribution
from collections import Counter
counts = Counter(y_combined)
print(f"\nClass distribution:")
for idx, count in sorted(counts.items()):
    print(f"  {all_labels[idx]}: {count}")