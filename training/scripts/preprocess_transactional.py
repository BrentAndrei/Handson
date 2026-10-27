#!/usr/bin/env python3
"""
Convert Transactional FSL dataset to our training format.
Extracts right-hand landmarks (63 features) to match our BiGRU model.
"""

import os
import numpy as np
import json
from pathlib import Path

# Input/output paths
INPUT_ROOT = "training/data/transactional_fsl/Transactional Filipino Sign Language Dataset/Transactional Filipino Sign Language Dataset/recorded_data"
OUTPUT_DIR = "training/data/processed_transactional"

WINDOW_SIZE = 16
STRIDE = 8

# Label encoder for 26 classes (alphabetical)
CLASSES = sorted([
    'AGAIN', 'CARD', 'CASH', 'COIN', 'DISCOUNT', 'EIGHT', 'FIVE', 'FOUR',
    'HELLO', 'HOW_MANY', 'HOW_MUCH', 'NINE', 'NO', 'ONE', 'PLEASE', 'PROBLEM',
    'RECEIPT', 'SEVEN', 'SIX', 'TEN', 'THANK_YOU', 'THREE', 'TWO', 'WAIT',
    'WELCOME', 'YES'
])
LABEL_ENCODER = {cls: i for i, cls in enumerate(CLASSES)}

def main():
    root = Path("training/data/transactional_fsl/Transactional Filipino Sign Language Dataset/Transactional Filipino Sign Language Dataset/recorded_data")
    
    print(f"Processing {len(CLASSES)} classes...")
    
    all_windows = []
    all_labels = []
    
    for cls in CLASSES:
        cls_dir = os.path.join("training/data/transactional_fsl/Transactional Filipino Sign Language Dataset/Transactional Filipino Sign Language Dataset/recorded_data", cls)
        if not os.path.isdir(cls_dir):
            continue
            
        npy_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        print(f"  {cls}: {len(npy_files)} samples")
        
        for npy_file in sorted(os.listdir(cls_dir)):
            if not npy_file.endswith('.npy'):
                continue
                
            data = np.load(os.path.join(cls_dir, npy_file))  # (T, 225)
            T, feat = data.shape
            if feat != 225 or len(data) < 16:
                continue
            
            # Extract right hand: landmarks 54-74 (indices 54-74 in 75 landmarks)
            # Each landmark has 3 coords, so right hand starts at 54*3=162, ends at 75*3=225
            right_hand_data = data[:, 162:225]  # (T, 63)
            
            # Sliding window
            stride = 8
            for i in range(0, len(right_hand_data) - 16 + 1, 8):
                window = right_hand_data[i:i+16]
                if window.shape == (16, 63):
                    all_windows.append(window.astype(np.float32))
                    all_labels.append(cls)
    
    if not all_windows:
        print("No windows generated!")
        return
    
    X = np.array(all_windows, dtype=np.float32)
    y = np.array([LABEL_ENCODER[l] for l in all_labels], dtype=np.int32)
    
    print(f"\nTotal windows: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(set(all_labels))}")
    
    # Save
    os.makedirs("training/data/processed_transactional", exist_ok=True)
    np.save("training/data/processed_transactional/X.npy", X)
    np.save("training/data/processed_transactional/y.npy", np.array([LABEL_ENCODER[l] for l in all_labels], dtype=np.int32))
    
    label_map = {str(v): k for k, v in {cls: i for i, cls in enumerate(CLASSES)}.items()}
    with open("training/data/processed_transactional/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\nSaved to training/data/processed_transactional/")
    print(f"X: {X.shape}, y: {len(all_labels)}")
    print(f"Classes: {CLASSES}")

if __name__ == "__main__":
    main()