#!/usr/bin/env python3
"""
Process Transactional FSL MP4 videos through MediaPipe Tasks API to extract right-hand landmarks.
Converts 1107 MP4s (26 classes) into training windows (16 frames, 63 features).
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Config
INPUT_ROOT = "training/data/transactional_fsl/Transactional Filipino Sign Language Dataset/Transactional Filipino Sign Language Dataset/recorded_data"
OUTPUT_DIR = "training/data/processed_transactional_video"
WINDOW_SIZE = 16
STRIDE = 8

FEATURE_DIM = 63  # 21 * 3

CLASSES = sorted([
    'AGAIN', 'CARD', 'CASH', 'COIN', 'DISCOUNT', 'EIGHT', 'FIVE', 'FOUR',
    'HELLO', 'HOW_MANY', 'HOW_MUCH', 'NINE', 'NO', 'ONE', 'PLEASE', 'PROBLEM',
    'RECEIPT', 'SEVEN', 'SIX', 'TEN', 'THANK_YOU', 'THREE', 'TWO', 'WAIT',
    'WELCOME', 'YES'
])

LABEL_ENCODER = {cls: i for i, cls in enumerate(CLASSES)}

# MediaPipe Hand Landmarker setup
MODEL_PATH = "training/data/hand_landmarker.task"

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
    if result.hand_landmarks:
        # Find right hand (MediaPipe returns both hands, we need to identify right)
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
    for i in range(0, len(frames_landmarks) - 16 + 1, 8):
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
    print(f"Processing Transactional FSL videos...")
    
    all_windows = []
    all_labels = []
    
    for cls in CLASSES:
        cls_dir = Path("training/data/transactional_fsl/Transactional Filipino Sign Language Dataset/Transactional Filipino Sign Language Dataset/recorded_data") / cls
        if not cls_dir.exists():
            continue
            
        mp4_files = sorted([f for f in cls_dir.iterdir() if f.suffix == '.mp4'])
        print(f"  {cls}: {len(mp4_files)} videos")
        
        for mp4_file in tqdm(sorted(cls_dir.glob('*.mp4')), desc=f"  {cls}", leave=False):
            try:
                windows = process_video(str(mp4_file))
                if len(windows) > 0:
                    all_windows.append(windows)
                    all_labels.extend([cls] * len(windows))
            except Exception as e:
                print(f"  Error processing {mp4_file.name}: {e}")
    
    if not all_windows:
        print("No windows generated!")
        return
    
    X = np.vstack(all_windows)
    y = np.array([CLASSES.index(l) for l in all_labels], dtype=np.int32)
    
    print(f"\nTotal windows: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(set(all_labels))}")
    
    # Save
    os.makedirs("training/data/processed_transactional_video", exist_ok=True)
    np.save("training/data/processed_transactional_video/X.npy", X)
    np.save("training/data/processed_transactional_video/y.npy", np.array([CLASSES.index(l) for l in all_labels], dtype=np.int32))
    
    label_map = {str(v): k for k, v in {cls: i for i, cls in enumerate(CLASSES)}.items()}
    with open("training/data/processed_transactional_video/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\nSaved to training/data/processed_transactional_video/")
    print(f"X: {X.shape}, y: {len(all_labels)}")

if __name__ == "__main__":
    import cv2
    import mediapipe as mp
    main()