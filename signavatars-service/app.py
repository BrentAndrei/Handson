"""
SMPL-X fitting service for sign language avatars.

Converts MediaPipe Holistic landmarks to SMPL-X pose parameters via a RESTful API.
Designed to sit behind the TypeScript/Three.js frontend as a hybrid backend.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from fit_smplx import SmplxFitter, MP_TO_SMPLX_JOINT_MAP

app = Flask(__name__)
CORS(app)

_model_dir = os.environ.get("SMPLX_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
_gender = os.environ.get("SMPLX_GENDER", "NEUTRAL")

# Lazily-loaded fitter singleton
_fitter: SmplxFitter | None = None

def get_fitter() -> SmplxFitter:
    global _fitter
    if _fitter is None:
        _fitter = SmplxFitter(_model_dir, _gender)
    return _fitter


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "signavatars-smplx"})


@app.route("/fit-frame", methods=["POST"])
def fit_frame():
    """Fit SMPL-X pose for a single frame of MediaPipe landmarks.

    Request JSON:
        {
            "landmarks": [x0,y0,z0, x1,y1,z1, ...],  // 345 floats
            "parent_world_quats": {"torso": [x,y,z,w], ...}  // optional
        }

    Response JSON:
        {
            "pose": [72 floats],
            "betas": [10 floats],
            "transl": [3 floats],
            "global_orient": [3 floats],
            "smplx_joints": [128 x 3 = 384 floats],
            "confidence": float
        }
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    landmarks_raw = data.get("landmarks")
    if landmarks_raw is None:
        return jsonify({"error": "'landmarks' field is required"}), 400

    landmarks = np.asarray(landmarks_raw, dtype=np.float64)
    if landmarks.size != 345:
        return jsonify({"error": f"Expected 345 landmark floats, got {landmarks.size}"}), 400

    landmarks = landmarks.reshape(115, 3)

    parent_world_quats = data.get("parent_world_quats")

    fitter = get_fitter()
    result = fitter.fit_frame(landmarks, parent_world_quats)

    return jsonify(result)


@app.route("/fit-sequence", methods=["POST"])
def fit_sequence():
    """Fit SMPL-X poses for a sequence of frames (batch mode).

    Request JSON:
        {
            "frames": [[x0,y0,z0, ...], [x0,y0,z0, ...], ...],
            "parent_world_quats": {...}  // optional, applied to all frames
        }

    Response JSON:
        {
            "frames": [{ "pose": [...], "betas": [...], ... }, ...],
            "label": "..."
        }
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    frames_raw = data.get("frames")
    if frames_raw is None:
        return jsonify({"error": "'frames' field is required"}), 400

    parent_world_quats = data.get("parent_world_quats")
    label = data.get("label", "unknown")

    fitter = get_fitter()
    results = []
    for frame_data in frames_raw:
        landmarks = np.asarray(frame_data, dtype=np.float64).reshape(115, 3)
        result = fitter.fit_frame(landmarks, parent_world_quats)
        results.append(result)

    return jsonify({"label": label, "frames": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
