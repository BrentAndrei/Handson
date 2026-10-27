#!/usr/bin/env python3
"""
Process WLASL pose data (184 classes) into training windows.
Extracts upper-body pose (7 joints * 3 = 21 features per frame).
Sliding window: 16 frames, stride 8.
"""

import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Config
PARQUET_PATH = "training/data/wlasl_pose.parquet"
WINDOW_SIZE = 16
STRIDE = 8
FEATURE_DIM = 21  # 7 joints * 3 coords

# SMPL-X upper body joints (indices in the 7-joint pose)
# 0: nose, 1: left_shoulder, 2: right_shoulder, 3: left_elbow, 4: right_elbow, 5: left_wrist, 6: right_wrist
UPPER_BODY_JOINTS = 7
FEATURE_DIM = 7 * 3  # 21

def load_wlasl_pose_data(parquet_path, window_size=16, stride=8):
    """Load WLASL pose data with sliding window."""
    df = pd.read_parquet(parquet_path)
    
    # Encode labels
    labels = sorted(df['label'].unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    X, y = [], []
    
    for idx, row in df.iterrows():
        poses = row['poses']  # Object array of (7, 3) arrays
        label = label_to_idx[row['label']]
        
        T = len(poses)
        if T < 16:
            continue
            
        # Stack frames to (T, 21) - 7 joints * 3 coords
        poses_flat = np.stack([np.stack(p).reshape(-1) for p in poses])  # (T, 21)
        
        # Sliding window
        stride = 8
        for i in range(0, len(poses) - 16 + 1, stride):
            window = poses[i:i+16]
            if len(window) == 16:
                window_flat = np.stack([np.stack(w).reshape(-1) for w in window])  # (16, 21)
                X.append(window_flat)
                y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    labels = sorted(set(df['label']))
    print(f"Total windows: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(labels)}")
    
    return X, y, labels

def main():
    print("Loading WLASL pose dataset...")
    X, y, labels = load_wlasl_pose_data("training/data/wlasl_pose.parquet")
    
    # Save processed data
    os.makedirs("training/data/processed_wlasl_pose", exist_ok=True)
    np.save("training/data/processed_wlasl_pose/X.npy", X)
    np.save("training/data/processed_wlasl_pose/y.npy", np.array(y, dtype=np.int32))
    
    label_map = {str(v): k for k, v in {label: i for i, label in enumerate(labels)}.items()}
    with open("training/data/processed_wlasl_pose/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\nSaved to training/data/processed_wlasl_pose/")
    print(f"X: {X.shape}, y: {len(y)}")
    print(f"Classes: {labels}")

if __name__ == "__main__":
    main()