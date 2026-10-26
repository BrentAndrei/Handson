#!/usr/bin/env python3
"""
Merge avatar-data JSON files (per-label, full sequences) with video_dataset.json
into a single combined dataset for training.
"""
import json
import os
import glob
from pathlib import Path
import numpy as np

# Constants from train.py / featureSelect.ts
FEATURE_DIM = 345
WINDOW_SIZE = 16
STRIDE = 6

# Paths
AVATAR_DATA_DIR = Path("public/avatar-data")
VIDEO_DATASET = Path("training/video_dataset.json")
OUTPUT_PATH = Path("training/combined_dataset.json")

def window_sequence(frames_flat, window_size=WINDOW_SIZE, stride=STRIDE, feature_dim=FEATURE_DIM):
    """Convert flat frame sequence into overlapping windows."""
    n_frames = len(frames_flat) // feature_dim
    if n_frames < window_size:
        return []
    windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        window = frames_flat[start * feature_dim : (start + window_size) * feature_dim]
        windows.append(window)
    return windows

def load_avatar_data():
    """Load all per-label JSON files from public/avatar-data/"""
    samples = []
    labels_set = set()
    
    json_files = list(AVATAR_DATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} avatar-data files")
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            label = data.get("label")
            frames_flat = data.get("frames", [])
            
            if not label or not frames_flat:
                print(f"  Skipping {json_file.name}: missing label or frames")
                continue
            
            # Convert to windows
            windows = window_sequence(frames_flat)
            if not windows:
                print(f"  Skipping {label}: only {len(frames_flat)//FEATURE_DIM} frames (< {WINDOW_SIZE})")
                continue
            
            for w in windows:
                samples.append({
                    "label": label,
                    "clip_id": f"avatar_{json_file.stem}",
                    "frames": w
                })
            labels_set.add(label)
            print(f"  {label}: {len(windows)} windows from {len(frames_flat)//FEATURE_DIM} frames")
            
        except Exception as e:
            print(f"  Error loading {json_file.name}: {e}")
    
    return samples, labels_set

def load_video_dataset():
    """Load the video dataset"""
    with open(VIDEO_DATASET) as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    labels = set(data.get("labels", []))
    
    print(f"Video dataset: {len(samples)} windows, {len(labels)} labels")
    return samples, labels

def main():
    print("Loading avatar-data...")
    avatar_samples, avatar_labels = load_avatar_data()
    
    print("\nLoading video dataset...")
    video_samples, video_labels = load_video_dataset()
    
    # Merge
    all_samples = avatar_samples + video_samples
    all_labels = sorted(avatar_labels | video_labels)
    
    print(f"\n=== MERGED ===")
    print(f"Total samples: {len(all_samples)}")
    print(f"Total labels: {len(all_labels)}")
    
    # Count per label
    from collections import Counter
    label_counts = Counter(s["label"] for s in all_samples)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Create combined dataset
    combined = {
        "featureDim": FEATURE_DIM,
        "windowSize": WINDOW_SIZE,
        "stride": STRIDE,
        "labels": all_labels,
        "samples": all_samples
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(combined, f)
    
    print(f"\nWritten to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()