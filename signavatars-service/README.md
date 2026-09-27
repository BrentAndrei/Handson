# SignAvatars SMPL-X Service

Python backend for fitting SMPL-X pose parameters from MediaPipe Holistic landmarks.
Designed as a hybrid companion to the TypeScript/Three.js frontend.

## Architecture

```
                    ┌─────────────────────────────┐
                    │  TypeScript Frontend         │
                    │  (Three.js + Vite + React)   │
                    │                             │
                    │  - Loads SMPL-X or MP data  │
                    │  - Renders Xbot avatar      │
                    │  - smplxConverter.ts        │
                    └──────────┬───────────────────┘
                               │  HTTP REST API
                    ┌──────────▼──────────────────┐
                    │  Python SMPL-X Service       │
                    │  (Flask + SMPL-X + NumPy)    │
                    │                             │
                    │  - app.py  (API endpoints)   │
                    │  - fit_smplx.py (fitter)     │
                    └─────────────────────────────┘
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download SMPL-X model files
#    Place SMPL-X model directories in models/ directory
#    If not present, the service falls back to IK-based fitting (no model needed)

# 4. Run the service
python app.py
```

The service runs on `http://localhost:8000` by default.

## API Endpoints

### GET /health
Returns service health status.

### POST /fit-frame
Fit SMPL-X parameters for a single frame.

**Request:**
```json
{
  "landmarks": [x0,y0,z0, x1,y1,z1, ...],  // 345 floats (115 landmarks × 3)
  "parent_world_quats": {"torso": [x,y,z,w], ...}  // optional
}
```

**Response:**
```json
{
  "pose": [72 floats],
  "betas": [10 floats],
  "transl": [3 floats],
  "global_orient": [3 floats],
  "smplx_joints": [384 floats],
  "confidence": 0.95
}
```

### POST /fit-sequence
Fit SMPL-X parameters for a sequence of frames (batch mode).

**Request:**
```json
{
  "frames": [[345 floats], [345 floats], ...],
  "parent_world_quats": {...},
  "label": "A"
}
```

**Response:**
```json
{
  "label": "A",
  "frames": [{ "pose": [...], "betas": [...], ... }, ...]
}
```

## Batch Conversion

Convert existing avatar data to SMPL-X format:

```bash
python convert_avatar_data.py \
  --input-dir  ../public/avatar-data \
  --output-dir ../public/avatar-data-smplx
```

## Fallback Mode

If SMPL-X model files are not available in the `models/` directory,
the service automatically falls back to a CCDIK-based inverse kinematics
solver. This produces pose parameters that approximate the MediaPipe
landmark positions without requiring the 500MB SMPL-X model download.

## Landmark Format

Each frame is 345 floats = 115 landmarks × 3 coordinates:

| Range      | Landmarks              | Count |
|------------|------------------------|-------|
| 0..98      | Pose (33)              | 33    |
| 99..161    | Left hand (21)         | 21    |
| 162..224   | Right hand (21)        | 21    |
| 225..344   | Face subset (40)       | 40    |
| **Total**  |                        | **115** |

Landmarks are normalized to the shoulder midpoint with shoulder-width scaling.
Coordinate system: MediaPipe (X=right, Y=down, Z=inward).
