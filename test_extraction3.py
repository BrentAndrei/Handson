import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import os
import mediapipe as mp

# Quick test - try IMAGE mode
model_path = 'training/holistic_landmarker.task'
print(f'Model exists: {os.path.exists(model_path)}')

options = vision.HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,  # Try IMAGE mode
    min_pose_detection_confidence=0.3,
    min_pose_landmarks_confidence=0.3,
    min_face_detection_confidence=0.5,
    min_face_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.3,
)
landmarker = vision.HolisticLandmarker.create_from_options(options)
print('Landmarker created successfully (IMAGE mode)')

# Test one frame - try with letterbox to 640x640
cap = cv2.VideoCapture('training/youtube_raw/A/36GlmDTYs6s.mp4')
ok, frame = cap.read()
cap.release()
print(f'Original frame shape: {frame.shape if ok else "FAILED"}')

if ok:
    # Try letterbox like image mode
    h, w = frame.shape[:2]
    size = 640
    scale = size / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    frame = canvas
    print(f'Letterbox frame shape: {frame.shape}')

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)  # IMAGE mode uses detect()
    print(f'Pose landmarks: {len(result.pose_landmarks) if result.pose_landmarks else 0}')
    print(f'Left hand: {len(result.left_hand_landmarks) if result.left_hand_landmarks else 0}')
    print(f'Right hand: {len(result.right_hand_landmarks) if result.right_hand_landmarks else 0}')
    print(f'Face: {len(result.face_landmarks) if result.face_landmarks else 0}')

landmarker.close()