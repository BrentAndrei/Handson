#!/usr/bin/env python3
"""Merge avatar-data JSON files with video_dataset.json - quiet version"""
import json
import os
from pathlib import Path
import numpy as np
from collections import Counter

FEATURE_DIM = 345
WINDOW_SIZE = 16
STRIDE = 6

AVATAR_DATA_DIR = Path("public/avatar-data")
VIDEO_DATASET = Path("training/video_dataset.json")
OUTPUT_PATH = Path("training/combined_dataset.json")

def window_sequence(frames_flat, window_size=WINDOW_SIZE, stride=STRIDE, feature_dim=FEATURE_DIM):
    n_frames = len(frames_flat) // feature_dim
    if n_frames < window_size:
        return []
    windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        window = frames_flat[start * feature_dim : (start + window_size) * feature_dim]
        windows.append(window)
    return windows

def load_avatar_data():
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
                continue
            windows = window_sequence(frames_flat)
            if not windows:
                continue
            for w in windows:
                samples.append({"label": label, "clip_id": f"avatar_{json_file.stem}", "frames": w})
            labels_set.add(label)
        except Exception:
            pass
    return samples, labels_set

def load_video_dataset():
    with open(VIDEO_DATASET) as f:
        data = json.load(f)
    return data.get("samples", []), set(data.get("labels", []))

def main():
    print("Loading avatar-data...")
    avatar_samples, avatar_labels = load_avatar_data()
    print(f"Avatar: {len(avatar_samples)} samples, {len(avatar_labels)} labels")
    
    print("Loading video dataset...")
    video_samples, video_labels = load_video_dataset()
    print(f"Video: {len(video_samples)} samples, {len(video_labels)} labels")
    
    all_samples = avatar_samples + video_samples
    all_labels = sorted(avatar_labels | video_labels)
    
    print(f"Merged: {len(all_samples)} samples, {len(all_labels)} labels")
    
    counts = Counter(s["label"] for s in all_samples)
    for label, count in sorted(counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {label}: {count}")
    
    combined = {
        "featureDim": FEATURE_DIM,
        "windowSize": WINDOW_SIZE,
        "stride": STRIDE,
        "labels": all_labels,
        "samples": all_samples
    }
    
    # Write in chunks to avoid memory issues
    with open(OUTPUT_PATH, "w") as f:
        f.write('{"featureDim":345,"windowSize":16,"stride":6,"labels":[')
        f.write(','.join(f'"{l}"' for l in all_labels))
        f.write('],"samples":[')
        for i, s in enumerate(all_samples):
            if i > 0:
                f.write(',')
            f.write(json.dumps(s, separators=(',', ':')))
        f.write(']}')
    
    print(f"Written to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()