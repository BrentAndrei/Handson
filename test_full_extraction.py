import cv2
import numpy as np
import os
import sys
sys.path.insert(0, 'training')

from extract_landmarks import make_landmarker, extract_clip_features, slice_into_windows

# Test with one video
video_path = 'training/youtube_raw/A/36GlmDTYs6s.mp4'
print(f'Testing: {video_path}')

landmarker = make_landmarker(use_gpu=False)
print('Landmarker created')

try:
    clip_features = extract_clip_features(video_path, landmarker)
    print(f'Extracted {len(clip_features)} frames')
    if clip_features:
        print(f'Feature dim: {len(clip_features[0])}')
        
    windows = slice_into_windows(clip_features)
    print(f'Generated {len(windows)} windows')
    if windows:
        print(f'Window size: {len(windows[0])} floats')
finally:
    landmarker.close()