"""Quick test of SMPL-X fitting on a single avatar data file."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_smplx import SmplxFitter, TOTAL_FLOATS_PER_FRAME
import numpy as np

with open(os.path.join(os.path.dirname(__file__), "..", "public", "avatar-data", "A.json")) as f:
    data = json.load(f)

frames = data["frames"]
print(f'Label: {data["label"]}, frameCount: {data["frameCount"]}, total floats: {len(frames)}')
print(f'Floats per frame: {len(frames) // data["frameCount"]}')

fitter = SmplxFitter(model_dir=None, gender="NEUTRAL")
# frames is a flat array of 32430 floats = 94 frames x 345 floats each
frame0 = np.asarray(frames[0:345], dtype=np.float64)
result = fitter.fit_frame(frame0)
print(f'Pose length: {len(result.pose)}')
print(f'Pose first 6: {result.pose[:6]}')
print(f'Betas length: {len(result.betas)}')
print(f'Transl: {result.transl}')
print(f'Confidence: {result.confidence}')
print(f'SMPL-X joints count: {len(result.smplx_joints) // 3}')
print("SUCCESS: SMPL-X fitting works without full model")
