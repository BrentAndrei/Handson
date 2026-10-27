#!/usr/bin/env python3
"""
Merge Kaggle FSL (28 classes) + Transactional FSL (26 classes) datasets.
Total: 54 classes, 63 features (right hand only).
"""

import os
import numpy as np
import json

# Load Kaggle data
kaggle_X = np.load("training/data/processed/X.npy")
kaggle_y = np.load("training/data/processed/y.npy")
with open("training/data/processed/label_map.json") as f:
    kaggle_label_map = json.load(f)

# Load Transactional data
trans_X = np.load("training/data/processed_transactional/X.npy")
trans_y = np.load("training/data/processed_transactional/y.npy")
with open("training/data/processed_transactional/label_map.json") as f:
    trans_label_map = json.load(f)

print(f"Kaggle: {len(kaggle_X)} windows, {len(set(kaggle_y))} classes")
print(f"Transactional: {len(trans_X)} windows, {len(set(trans_y))} classes")

# Check for overlapping labels
kaggle_labels = set([kaggle_label_map[str(i)] for i in range(len(kaggle_label_map))])
trans_labels = set([trans_label_map[str(i)] for i in range(len(trans_label_map))])
print(f"Kaggle labels: {sorted(kaggle_labels)}")
print(f"Trans labels: {sorted(trans_labels)}")
print(f"Overlap: {kaggle_labels & trans_labels}")

# Build unified label map
all_labels = sorted(list(kaggle_labels | trans_labels))
label_to_idx = {label: i for i, label in enumerate(all_labels)}
print(f"\nTotal unique labels: {len(all_labels)}")
print(f"Labels: {all_labels}")

# Remap labels
kaggle_y_new = np.array([label_to_idx[kaggle_label_map[str(l)]] for l in kaggle_y], dtype=np.int32)
trans_y_new = np.array([label_to_idx[trans_label_map[str(l)]] for l in trans_y], dtype=np.int32)

# Combine
X_combined = np.concatenate([kaggle_X, trans_X], axis=0)
y_combined = np.concatenate([kaggle_y_new, trans_y_new], axis=0)

print(f"\nCombined: {len(X_combined)} windows, {len(set(y_combined))} classes")
print(f"Shape: {X_combined.shape}")

# Save combined dataset
os.makedirs("training/data/processed_combined", exist_ok=True)
np.save("training/data/processed_combined/X.npy", X_combined)
np.save("training/data/processed_combined/y.npy", y_combined)

label_map = {str(i): label for i, label in enumerate(all_labels)}
with open("training/data/processed_combined/label_map.json", 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"\nSaved to training/data/processed_combined/")
print(f"Total classes: {len(all_labels)}")
print(f"Total samples: {len(X_combined)}")

# Verify class distribution
from collections import Counter
counts = Counter(y_combined)
print(f"\nClass distribution:")
for idx, count in sorted(counts.items()):
    print(f"  {all_labels[idx]}: {count}")