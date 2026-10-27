#!/usr/bin/env python3
"""
Process FSL-105 dataset (105 classes) - extract right-hand landmarks from MOV videos.
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# Config
INPUT_ROOT = "training/data/fsl105/FSL-105 A dataset for recognizing 105 Filipino sign language videos"
OUTPUT_DIR = "training/data/processed_fsl105"
WINDOW_SIZE = 16
STRIDE = 8

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "training/data/hand_landmarker.task"

# Config
CLASSES_FILE = "training/data/fsl105/FSL-105 A dataset for recognizing 105 Filipino sign language videos/labels.csv"

# MediaPipe Hand Landmarker setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_right_hand_landmarks(result):
    """Extract right hand landmarks (63 features) from MediaPipe results."""
    if not result.hand_landmarks or not result.handedness:
        return np.zeros(63, dtype=np.float32)
    
    for i, handedness in enumerate(result.handedness):
        if handedness[0].category_name == "Right":
            landmarks = result.hand_landmarks[i]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            return coords.flatten()  # 63 features
    return np.zeros(63, dtype=np.float32)

def process_video(video_path, window_size=16, stride=8):
    """Extract right-hand landmarks from a video file using MediaPipe HandLandmarker."""
    # Create hand landmarker
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="training/data/hand_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    frames_landmarks = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Calculate timestamp in milliseconds
        timestamp_ms = int(frame_idx * 1000 / fps)
        
        # Process frame
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract right hand landmarks
        landmarks = extract_right_hand_landmarks(result)
        frames_landmarks.append(landmarks)
        frame_idx += 1
    
    cap.release()
    
    if len(frames_landmarks) < 16:
        return []
    
    # Sliding window
    windows = []
    for i in range(0, len(frames_landmarks) - 16 + 1, stride):
        window = frames_landmarks[i:i+16]
        if len(window) == 16:
            windows.append(window)
    
    return np.array(windows, dtype=np.float32)

def extract_right_hand_landmarks(result):
    """Extract right hand landmarks (63 features) from MediaPipe results."""
    if not result.hand_landmarks or not result.handedness:
        return np.zeros(63, dtype=np.float32)
    
    for i, handedness in enumerate(result.handedness):
        if handedness[0].category_name == "Right":
            landmarks = result.hand_landmarks[i]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            return coords.flatten()  # 63 features
    return np.zeros(63, dtype=np.float32)

def main():
    print("Loading FSL-105 labels...")
    labels_df = pd.read_csv("training/data/fsl105/FSL-105 A dataset for recognizing 105 Filipino sign language videos/labels.csv")
    labels = sorted(labels_df['label'].unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    print(f"Classes: {len(labels)}")
    print(f"Labels: {labels}")
    
    # Load train CSV
    train_df = pd.read_csv("training/data/fsl105/FSL-105 A dataset for recognizing 105 Filipino sign language videos/train.csv")
    print(f"Total train videos: {len(train_df)}")
    
    all_windows = []
    all_labels = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing videos"):
        cls = row['label']
        video_rel = row['vid_path']
        video_path = os.path.join("training/data/fsl105/FSL-105 A dataset for recognizing 105 Filipino sign language videos", video_rel)
        
        if not os.path.exists(video_path):
            continue
            
        try:
            windows = process_video(video_path)
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.extend([cls] * len(windows))
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
    
    if not all_windows:
        print("No windows generated!")
        return
    
    X = np.vstack(all_windows)
    y = np.array([labels.index(l) for l in all_labels], dtype=np.int32)
    
    print(f"\nTotal windows: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(set([labels[l] for l in all_labels]))}")
    
    # Save
    os.makedirs("training/data/processed_fsl105_video", exist_ok=True)
    np.save("training/data/processed_fsl105_video/X.npy", X)
    np.save("training/data/processed_fsl105_video/y.npy", np.array([labels.index(l) for l in all_labels], dtype=np.int32))
    
    label_map = {str(v): k for k, v in {cls: i for i, cls in enumerate(sorted(labels))}.items()}
    with open("training/data/processed_fsl105_video/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\nSaved to training/data/processed_fsl105_video/")
    print(f"X: {X.shape}, y: {len(all_labels)}")
    print(f"Classes: {len(labels)}")

if __name__ == "__main__":
    import cv2
    import mediapipe as mp
    main()