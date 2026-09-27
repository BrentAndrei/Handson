# AGENTS.md - SignAvatar Hybrid Migration

## Project Overview

Handson is a sign language avatar project using:
- **Frontend**: TypeScript + Three.js + React + Vite
- **Avatar model**: Xbot (GLB), loaded via GLTFLoader
- **Landmarks**: MediaPipe Holistic (pose 33, left hand 21, right hand 21, face 40)
- **Avatar data**: Pre-recorded sign sequences as JSON in `public/avatar-data/`

## Architecture

### Current (Quaternion-Based) Pipeline

```
MediaPipe Holistic → Landmarks (345 floats/frame) → normalizeLandmarks.ts
  → dataLoader.ts → retarget.ts (quaternion alignment)
  → SignAvatar.tsx (Three.js bone application)
```

### Hybrid Migration Pipeline (target)

```
MediaPipe Holistic → Landmarks (345 floats/frame) → signavatars-service (Python)
  → SMPL-X pose parameters (72 floats) → smplxConverter.ts
  → SignAvatar.tsx (Three.js bone application)
```

The Python service (signavatars-service/) handles SMPL-X fitting using:
1. Full SMPL-X model (requires model files) — `smplx` library
2. IK fallback (no model needed) — CCDIK solver

## Key Files

### TypeScript Frontend
| File | Purpose |
|------|---------|
| `src/avatar/retarget.ts` | Quaternion-based bone alignment from MediaPipe landmarks |
| `src/avatar/smplxConverter.ts` | SMPL-X pose params → AvatarFrame converter |
| `src/avatar/dataLoader.ts` | Data loading with SMPL-X service fallback |
| `src/components/SignAvatar.tsx` | Three.js renderer, bone animation, idle poses |
| `src/mediapipe/types.ts` | Landmark layout constants (345 floats per frame) |
| `src/mediapipe/pack.ts` | Pack HolisticLandmarkerResult into flat buffer |
| `src/utils/normalizeLandmarks.ts` | Shoulder-midpoint normalization |

### Python Service
| File | Purpose |
|------|---------|
| `signavatars-service/app.py` | Flask REST API endpoints |
| `signavatars-service/fit_smplx.py` | SMPL-X fitting logic, IK solver |
| `signavatars-service/convert_avatar_data.py` | Batch convert existing data |
| `signavatars-service/requirements.txt` | Python dependencies |

### Avatar Model (Xbot GLB)
- Model: `public/models/Xbot.glb`
- Bone structure: mixamorigX_Hips/Spine/Neck/Head, LeftArm/ForeArm/Hand, etc.
- Bones mapped via `BONE_MAP` in `SignAvatar.tsx`

## SMPL-X Joint Mapping

SMPL-X body joints (72-dim pose):
- Indices 0-2: global_orient (root rotation, axis-angle)
- Indices 3-65: body_pose (21 joints × 3 axis-angle)
- Indices 66-71: hand_pose (3 per hand, axis-angle)

SMPL-X joint indices → Xbot bone mapping (in `smplxConverter.ts`):
- 0 (pelvis) → root
- 9 (spine_3) → torso
- 12 (neck) → neck
- 15 (head) → head
- 13 (left_clavicle) → leftShoulder
- 14 (right_clavicle) → rightShoulder
- 16 (left_upper_arm) → leftUpperArm
- 17 (right_upper_arm) → rightUpperArm
- 20 (left_forearm) → leftForearm
- 21 (right_forearm) → rightForearm
- 22 (left_hand) → leftHand
- 23 (right_hand) → rightHand

## Configuration

### Enabling SMPL-X Service
```typescript
import { enableSmplxService, setSmplxServiceUrl } from './avatar/dataLoader';
enableSmplxService(true);
setSmplxServiceUrl('http://localhost:8000');
```

### Environment Variables
- `VITE_SMPLX_SERVICE_URL`: Override the default service URL (default: `http://localhost:8000`)
- `SMPLX_MODEL_DIR`: Python path to SMPL-X model files
- `SMPLX_GENDER`: NEUTRAL, MALE, or FEMALE

## Migration Phases

- **Phase 0 (DONE)**: Backup to GitHub (tag `v0.1-backup-quaternion-retargeting`)
- **Phase 1 (DONE)**: Python SMPL-X service (app.py, fit_smplx.py)
- **Phase 2 (DONE)**: TypeScript converter (smplxConverter.ts)
- **Phase 3 (DONE)**: Data loader integration with fallback
- **Phase 4 (TODO)**: Batch convert existing avatar data
- **Phase 5 (TODO)**: Enable SMPL-X as default, keep quaternion as fallback

## Testing

```bash
# TypeScript type check
npx tsc --noEmit

# Run dev server (port 5174)
npm run dev

# Python service
cd signavatars-service && python app.py
```
