import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import os

# Quick test
model_path = 'training/holistic_landmarker.task'
print(f'Model exists: {os.path.exists(model_path)}')

options = vision.HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.3,
    min_pose_landmarks_confidence=0.3,
    min_face_detection_confidence=0.5,
    min_face_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.3,
)
landmarker = vision.HolisticLandmarker.create_from_options(options)
print('Landmarker created successfully')

# Test one frame
cap = cv2.VideoCapture('training/youtube_raw/A/36GlmDTYs6s.mp4')
ok, frame = cap.read()
cap.release()
print(f'Frame shape: {frame.shape if ok else "FAILED"}')

if ok:
    import mediapipe as mp
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, 0)
    print(f'Pose landmarks: {len(result.pose_landmarks) if result.pose_landmarks else 0}')
    print(f'Left hand: {len(result.left_hand_landmarks) if result.left_hand_landmarks else 0}')
    print(f'Right hand: {len(result.right_hand_landmarks) if result.right_hand_landmarks else 0}')
    print(f'Face: {len(result.face_landmarks) if result.face_landmarks else 0}')

landmarker.close()