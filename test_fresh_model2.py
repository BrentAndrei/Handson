import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import os
import mediapipe as mp

# Try fresh model
model_path = 'training/holistic_landmarker_new.task'
print(f'Model exists: {os.path.exists(model_path)}')

options = vision.HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.1,
    min_pose_landmarks_confidence=0.1,
    min_face_detection_confidence=0.1,
    min_face_landmarks_confidence=0.1,
    min_hand_landmarks_confidence=0.1,
)
landmarker = vision.HolisticLandmarker.create_from_options(options)
print('Landmarker created')

# Use a test image from candidates
test_img_path = 'scripts/fixtures/poses/candidates/tpose_0.jpg'
frame = cv2.imread(test_img_path)
print(f'Test image shape: {frame.shape}')

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
result = landmarker.detect(mp_image)
print(f'Pose landmarks: {len(result.pose_landmarks) if result.pose_landmarks else 0}')
if result.pose_landmarks:
    for i, lm in enumerate(result.pose_landmarks[0][:5]):
        print(f'  {i}: x={lm.x:.3f} y={lm.y:.3f} z={lm.z:.3f}')
print(f'Left hand: {len(result.left_hand_landmarks) if result.left_hand_landmarks else 0}')
print(f'Right hand: {len(result.right_hand_landmarks) if result.right_hand_landmarks else 0}')
print(f'Face: {len(result.face_landmarks) if result.face_landmarks else 0}')

landmarker.close()