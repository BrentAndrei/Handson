#!/usr/bin/env python3
"""
Merge 54-class combined (Kaggle + Transactional) + WLASL pose (184) datasets.
Total: 54 + 184 = 238 classes.
Hand: 63 features (right hand), Pose: 21 features (7 joints).
Need to handle different feature dimensions.
"""

import os
import json
import numpy as np

# Load 54-class combined data (hand only, 63 features)
hand_X = np.load("training/data/processed_all/X.npy")  # (16850, 16, 63)
hand_y = np.load("training/data/processed_all/y.npy")
with open("training/data/processed_all/label_map.json") as f:
    hand_label_map = json.load(f)

# Load WLASL pose data (184 classes, 21 features)
pose_X = np.load("training/data/processed_wlasl_pose/X.npy")  # (7914, 16, 21)
pose_y = np.load("training/data/processed_wlasl_pose/y.npy")
with open("training/data/processed_wlasl_pose/label_map.json") as f:
    pose_label_map = json.load(f)

print(f"Hand data: {len(hand_X)} windows, {len(set(hand_y))} classes")
print(f"Pose data: {len(pose_X)} windows, {len(set(pose_y))} classes")

# Check labels
hand_labels = set([hand_label_map[str(i)] for i in range(len(hand_label_map))])
pose_labels = set([pose_label_map[str(i)] for i in range(len(pose_label_map))])
print(f"Hand labels: {len(hand_labels)}")
print(f"Pose labels: {len(pose_labels)}")
print(f"Overlap: {hand_labels & pose_labels}")

# No overlap expected - different vocabularies
# Total unique labels: 54 + 184 = 238
all_labels = sorted(list(hand_labels | pose_labels))
label_to_idx = {label: i for i, label in enumerate(all_labels)}
print(f"\nTotal unique labels: {len(all_labels)}")

# Remap labels
hand_y_new = np.array([label_to_idx[hand_label_map[str(l)]] for l in hand_y], dtype=np.int32)
pose_y_new = np.array([label_to_idx[pose_label_map[str(l)]] for l in pose_y], dtype=np.int32)

# Pad pose data to 63 features (pad with zeros)
# Pose has 21 features, hand has 63. Pad pose to 63.
pose_X_padded = np.zeros((len(pose_X), 16, 63), dtype=np.float32)
pose_X_padded[:, :, :21] = pose_X  # First 21 features are pose

# Combine
X_combined = np.concatenate([hand_X, pose_X_padded], axis=0)
y_combined = np.concatenate([hand_y_new, pose_y_new], axis=0)

print(f"\nFinal combined: {len(X_combined)} windows, {len(set(y_combined))} classes")
print(f"Shape: {X_combined.shape}")

# Save combined dataset
os.makedirs("training/data/processed_all238", exist_ok=True)
np.save("training/data/processed_all238/X.npy", X_combined)
np.save("training/data/processed_all238/y.npy", y_combined)

label_map = {str(i): label for i, label in enumerate(all_labels)}
with open("training/data/processed_all238/label_map.json", 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"\nSaved to training/data/processed_all238/")
print(f"Total classes: {len(all_labels)}")
print(f"Total samples: {len(X_combined)}")

# Class distribution
from collections import Counter
counts = Counter(y_combined)
print(f"\nClass distribution (first 20):")
for idx, count in sorted(counts.items())[:20]:
    print(f"  {all_labels[idx]}: {count}")