import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import os
import mediapipe as mp

# Try to create a simple test image with a person pose to test the model
model_path = 'training/holistic_landmarker.task'
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

# Create a simple test - use a webcam frame or existing image if available
# Let's try using the test image from the verify tests
test_img_path = 'scripts/fixtures/poses/pose_A.png'
if os.path.exists(test_img_path):
    frame = cv2.imread(test_img_path)
    print(f'Test image shape: {frame.shape}')
else:
    print('Test image not found, creating synthetic...')
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a simple stick figure
    cv2.circle(frame, (320, 150), 50, (255, 255, 255), -1)  # head
    cv2.line(frame, (320, 200), (320, 400), (255, 255, 255), 10)  # spine
    cv2.line(frame, (320, 250), (200, 350), (255, 255, 255), 10)  # left arm
    cv2.line(frame, (320, 250), (440, 350), (255, 255, 255), 10)  # right arm
    cv2.line(frame, (320, 400), (200, 550), (255, 255, 255), 10)  # left leg
    cv2.line(frame, (320, 400), (440, 550), (255, 255, 255), 10)  # right leg

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
result = landmarker.detect(mp_image)
print(f'Pose landmarks: {len(result.pose_landmarks) if result.pose_landmarks else 0}')
print(f'Left hand: {len(result.left_hand_landmarks) if result.left_hand_landmarks else 0}')
print(f'Right hand: {len(result.right_hand_landmarks) if result.right_hand_landmarks else 0}')
print(f'Face: {len(result.face_landmarks) if result.face_landmarks else 0}')

landmarker.close()