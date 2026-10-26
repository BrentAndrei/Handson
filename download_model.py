import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import urllib.request
import os

# Check available models
print('MediaPipe version:', mp.__version__)

# Try to download the model again
url = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task'
print('Downloading fresh model...')
urllib.request.urlretrieve(url, 'training/holistic_landmarker_new.task')
print('Done')
print(f'Size: {os.path.getsize("training/holistic_landmarker_new.task")} bytes')