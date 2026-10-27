#!/usr/bin/env python3
"""
Convert Kaggle FSL CSV dataset to numpy windows for training.
Each CSV = 1 frame (21 hand landmarks x 4 values = 84 features).
Group 100 frames = 1 sample. Sliding window = 16 frames.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import json

KAGGLE_DIR = "training/data/fsl_kaggle/media_pipe_keypoints_dataset"
OUTPUT_DIR = "training/data/processed"
WINDOW_SIZE = 16
STRIDE = 8
FRAMES_PER_SAMPLE = 100

def process_class(cls_dir, class_name):
    """Process all CSV files in a class directory."""
    csv_files = []
    for f in os.listdir(cls_dir):
        if f.endswith('.csv'):
            try:
                # Extract number from filename like "A_1.csv" or "E_1 2.csv"
                num_str = f.split('_')[1].split('.')[0].replace(' ', '')
                csv_files.append((int(num_str), f))
            except (ValueError, IndexError):
                continue
    csv_files.sort(key=lambda x: x[0])
    csv_files = [f for _, f in csv_files]
    
    print(f"  {cls_dir.name}: {len(csv_files)} frames")
    
    # Group frames into samples (100 frames each)
    samples = []
    for i in range(0, len(csv_files), 100):
        frame_files = csv_files[i:i+100]
        if len(frame_files) < 50:  # Skip incomplete samples
            continue
            
        sample_frames = []
        for f in frame_files:
            df = pd.read_csv(cls_dir / f, header=None, names=['landmark', 'x', 'y', 'z', 'vis'], skiprows=1)
            # Extract x,y,z for 21 landmarks = 63 features per frame
            coords = df[['x','y','z']].values.flatten().astype(np.float32)  # 21*3 = 63 features
            sample_frames.append(coords)
        
        if len(sample_frames) >= 50:  # At least 50 frames
            samples.append(np.array(sample_frames, dtype=np.float32))
    
    return samples

def sliding_windows(frames, window_size=16, stride=8):
    """Generate sliding windows from a sequence."""
    windows = []
    T = len(frames)
    for i in range(0, T - window_size + 1, stride):
        window = frames[i:i+window_size]
        if len(window) == window_size:
            windows.append(window)
    return np.array(windows)

def main():
    data_root = Path("training/data/fsl_kaggle/media_pipe_keypoints_dataset")
    classes = sorted([d for d in os.listdir("training/data/fsl_kaggle/media_pipe_keypoints_dataset") 
                      if os.path.isdir(os.path.join("training/data/fsl_kaggle/media_pipe_keypoints_dataset", d))])
    
    print(f"Processing {len(classes)} classes...")
    
    all_windows = []
    all_labels = []
    label_encoder = {cls: i for i, cls in enumerate(sorted(os.listdir("training/data/fsl_kaggle/media_pipe_keypoints_dataset")))}
    
    for cls in sorted(os.listdir("training/data/fsl_kaggle/media_pipe_keypoints_dataset")):
        cls_dir = Path("training/data/fsl_kaggle/media_pipe_keypoints_dataset") / cls
        if not cls_dir.is_dir():
            continue
            
        print(f"\nProcessing {cls}...")
        samples = process_class(Path(cls_dir), cls)
        print(f"  {len(samples)} valid samples")
        
        for sample_frames in samples:
            windows = sliding_windows(sample_frames, 16, 8)
            if len(windows) > 0:
                all_windows.append(windows)
                labels = [label_encoder[cls]] * len(windows)
                all_labels.extend(labels)
    
    if not all_windows:
        print("No windows generated!")
        return
    
    X = np.vstack(all_windows)
    y = np.array(all_labels)
    
    print(f"\nTotal windows: {len(X)}")
    print(f"Shape: {X.shape}")
    
    # Save processed data
    os.makedirs("training/data/processed", exist_ok=True)
    np.save("training/data/processed/X.npy", X)
    np.save("training/data/processed/y.npy", np.array(all_labels))
    
    # Save label map
    label_map = {str(v): k for k, v in label_encoder.items()}
    with open("training/data/processed/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\nSaved to training/data/processed/")
    print(f"X: {X.shape}, y: {len(all_labels)}")
    print(f"Classes: {list(label_encoder.keys())}")

if __name__ == "__main__":
    main()