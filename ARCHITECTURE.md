# SignAvatar System Architecture & Data Pipeline

Technical reference for the handson sign language avatar system.

---

## 1. System Architecture: End-to-End Workflow

The system operates in two parallel modes: **live camera sign recognition** and **pre-recorded sign playback**. Both converge at the Three.js rendering layer.

### 1.1 Component Map

```
┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐
│  Webcam (60fps) │─────▶│  App.tsx (React) │─────▶│  useHolisticPipeline │
└─────────────────┘      └──────────────────┘      └────────┬───────────┘
                                                              │
                                              HolisticLandmarkerResult
                                                              │
                              ┌───────────────────────────────┴───────────────┐
                              │                                               │
                    ┌─────────▼──────────┐                          ┌───────▼────────┐
                    │  Live Pipeline     │                          │  Recording     │
                    │  (Recognition)     │                          │  Pipeline      │
                    │                    │                          │                │
                    │ normalizeLandmarks │◀─ packHolisticResult      │ packHolistic   │
                    │  ↓                 │                          │  ↓             │
                    │ selectFeatures     │                          │ normalize      │
                    │  ↓                 │                          │  ↓             │
                    │ SlidingWindowBuf   │                          │ selectFeatures│
                    │  ↓ (16 frames)     │                          │  ↓             │
                    │ useSignInference   │                          │ store to JSON │
                    │  ↓ (TensorFlow.js) │                          └────────────────┘
                    │  prediction        │
                    └────────────────────┘
                              │
                              │ label triggers playback
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       SignAvatar.tsx (Three.js)                         │
│                                                                        │
│   Data Source Decision:                                                │
│   1. SMPL-X enabled?  →  dataLoader.ts → loadLabelDataSmplx()          │
│      → fetch /avatar-data-smplx/<label>.smplx.json                      │
│      → smplxConverter.ts → smplxToAvatarFrames() → AvatarFrame[]      │
│                                                                        │
│   2. Fallback:       →  dataLoader.ts → loadLabelData() (quaternion)   │
│      → fetch /avatar-data/<label>.json                                 │
│      → retarget.ts → retargetFrames() → smoothFramesSpline() → Avatar  │
│                                                                        │
│   Animation Engine:                                                    │
│   - GLTFLoader → Xbot.glb                                                │
│   - BONE_MAP maps mixamorigX_* → avatar keys                           │
│   - requestAnimationFrame loop applies bone.quaternion.slerp()        │
│   - Idle animation applied when not actively signing                    │
│   - Post-processing: EffectComposer + UnrealBloomPass + OutputPass      │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Runtime State Machine (SignAvatar.tsx)

- **LOADING**: Fetch manifests, instantiate WebGL renderer, load Xbot.glb
- **READY**: Model loaded, bones mapped, bind-pose quaternions captured, parent-world quaternions captured
- **PLAYING**: Animation loop interpolates between sequential `AvatarFrame` objects using `slerpQuaternions` with `SMOOTHING = 0.25` factor and `MAX_BONE_ANGLE = 162°` constraint
- **IDLE**: When not animating, applies sinusoidal micro-movements to head, neck, shoulders, arms, forearms, hands, and torso around bind-pose quaternions

### 1.3 Playback Engine

- Frames are **subsampled** to ~16 keyframes per sign for smooth interpolation
- Transition between keyframes: ease-in-out cubic, duration = `0.4 / playbackSpeed` seconds
- Quaternion SLERP between `prevFrameRef` and `targetFrameRef2` applied per bone each animation tick
- Bone quaternion angles exceeding `MAX_BONE_ANGLE` are interpolated with damping (`SMOOTHING * 0.5`)
- All bone quaternions renormalized every frame to prevent drift

---

## 2. Data Acquisition

### 2.1 Input Source: Webcam

- **Device**: HTML5 `<video>` element via `navigator.mediaDevices.getUserMedia()`
- **Resolution**: Inference downscaled to configurable size via offscreen canvas (default: original video dimensions)
- **Frame rate**: Target 24 FPS (`FRAME_INTERVAL_MS = 41.667ms`); accumulation buffer prevents faster-than-target detection

### 2.2 MediaPipe Holistic Landmarker

- **Model**: `holistic_landmarker` (MediaPipe Tasks v0.10.20)
- **Model asset**: `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`
- **Execution**: WebAssembly (CPU delegate), running mode: `VIDEO`
- **Detection thresholds**: All set to 0.7 (pose, pose presence, face, face presence, hand landmarks)

### 2.3 Output Landmark Sets

| Set | Count | Description |
|-----|-------|-------------|
| Pose | 33 landmarks | Full-body skeletal tracking |
| Left hand | 21 landmarks | Full hand skeleton |
| Right hand | 21 landmarks | Full hand skeleton |
| Face | 468 landmarks | Full face mesh (reduced to 40 for features) |

### 2.4 Coordinate System

- **MediaPipe native**: X = right, Y = down, Z = into screen (right-handed)
- **Converted to OpenGL/GLTF**: `mediapipeToGltf(v) = [v[0], -v[1], -v[2]]` — flips Y and Z to Y-up, Z-outward
- **Normalization**: Shoulder midpoint (landmarks 11 + 12) / 2 as anchor; shoulder-to-shoulder distance as scale factor
- **Normalized space**: `vector[i] = (source[i] - anchor) / scale`

---

## 3. Data Types and Formats

### 3.1 Raw Landmark Buffer (Frontend Internal)

- **Type**: `Float32Array` backed by `ArrayBuffer`
- **Size**: 345 floats per frame (115 landmarks × 3 coordinates)
- **Layout** (`src/mediapipe/types.ts`):

| Offset | Count | Points | Content |
|--------|-------|--------|---------|
| 0 | 33 | 99 floats | Pose landmarks (indices 0–32) |
| 99 | 21 | 63 floats | Left hand landmarks |
| 162 | 21 | 63 floats | Right hand landmarks |
| 225 | 40 | 120 floats | Face landmarks (lips subset) |

### 3.2 NormalizedFrame

```typescript
interface NormalizedFrame {
  vector: Float32Array;  // 345 floats, translated + scaled in place
  anchor: [number, number, number];   // shoulder midpoint in original coords
  scale: number;         // shoulder-to-shoulder distance
  usedFallbackScale: boolean;  // true if shoulder distance was degenerate
}
```

### 3.3 Feature Vector (for ML)

- **Source**: `selectFeatures()` reduces normalized vector to `FEATURE_DIM = (33 + 21 + 21 + 40) * 3 = 201 floats`
- **Selection**: Pose (33) + Left hand (21) + Right hand (21) + Lip contour (40 specific indices)
- **Lip indices**: `[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, ...]` (40 specific face mesh points)

### 3.4 Sliding Window Buffer

- **Size**: 16 frames (`WINDOW_SIZE`, stride 6)
- **Emission**: `shouldEmit()` returns true every 6 frames after buffer fills
- **Output**: `Float32Array(WINDOW_SIZE * FEATURE_DIM)` = 3216 floats, row-major

### 3.5 SMPL-X Pose Format

```typescript
interface SmplxPose {
  pose: number[];        // 72 floats [global_orient(3) + body_pose(63) + hand_pose(6)]
  betas: number[];       // 10 floats (shape parameters, all zeros in IK mode)
  transl: number[];      // 3 floats (root translation, zeroed in practice)
  global_orient: number[]; // 3 floats (axis-angle, zero in IK mode)
  smplx_joints: number[]; // 128 × 3 = 384 floats (joint positions)
  confidence: number;     // 0.0–1.0
}
```

**Pose breakdown**:
- `[0:3]` — `global_orient`: root rotation as axis-angle (always `[0, 0, 0]` in IK mode)
- `[3:66]` — `body_pose`: 21 joints × 3 axis-angle values (joints 1–21, root at index 0 excluded)
- `[66:69]` — `left hand_pose`: 3 axis-angle values encoding local hand orientation
- `[69:72]` — `right hand_pose`: 3 axis-angle values encoding local hand orientation

### 3.6 AvatarFrame (Engine-Ready Format)

```typescript
interface AvatarFrame {
  root: [number, number, number];      // position (zeroed in SMPL-X path)
  hips: [number, number, number];      // position (zeroed)
  torso: Quat;       // [x, y, z, w]
  head: Quat;
  neck: Quat;
  leftShoulder: Quat;
  leftUpperArm: Quat;
  leftForearm: Quat;
  leftHand: Quat;
  rightShoulder: Quat;
  rightUpperArm: Quat;
  rightForearm: Quat;
  rightHand: Quat;
}
// Quat = [number, number, number, number] (x, y, z, w)
```

### 3.7 Manifest Format

**Source manifest** (`public/avatar-data/manifest.json`):
```json
{ "labels": ["ADOPT", "AGAIN", ...], "entries": [{"label": "ADOPT", "file": "ADOPT.json", "smplxFile": "ADOPT.smplx.json", "frameCount": 79}] }
```

**SMPL-X manifest** (`public/avatar-data-smplx/manifest.json`):
```json
{ "totalLabels": 402, "totalFrames": 193152, "labels": ["ADOPT", ...], "entries": [{"label": "ADOPT", "file": "ADOPT.smplx.json", "frameCount": 79, "smplxFile": "ADOPT.smplx.json"}] }
```

### 3.8 Pre-recorded Avatar Data Format

**Quaternion-based** (`public/avatar-data/<label>.json`):
```json
{ "label": "ADOPT", "frameCount": 79, "frames": [[345 floats], [345 floats], ...] }
```

**SMPL-X** (`public/avatar-data-smplx/<label>.smplx.json`):
```json
{ "label": "ADOPT", "frameRate": 30, "frameCount": 79, "frames": [{"pose": [72 floats], "betas": [10], "transl": [3], "global_orient": [3], "smplx_joints": [384], "confidence": 0.95}, ...] }
```

---

## 4. Data Pipeline and Transformation

### 4.1 Live Camera Pipeline

```
Webcam → HolisticLandmarker → HolisticLandmarkerResult
  ↓ (packHolisticResult)
Packed Float32Array (345 floats, 115 points × 3)
  ↓ (normalizeLandmarks)
NormalizedFrame { vector, anchor, scale }
  ↓ (selectFeatures)
Feature Vector (201 floats) + SlidingWindowBuffer (16 frames)
  ↓ (useSignInference: tf.loadLayersModel + tf.tensor([1, 16, 201]))
Prediction { label, confidence }
  ↓
Trigger playback of <label>
```

#### 4.1.1 Smoothing (Display Only)

- **One Euro Filter** (`src/mediapipe/smoothing.ts`): Per-point low-pass filter with adaptive cutoff
  - Near-stationary points: aggressive filtering (kills jitter)
  - Fast-moving points: light filtering (preserves motion)
  - Applied only to canvas overlay drawing, not to the data pushed downstream
- **Temporal extrapolation**: When detection misses a frame, linearly extrapolates from velocity of previous detection (max 150ms hold)
- **Hold-fade**: Smoothed pose is held and faded when no new detection arrives

### 4.2 SMPL-X Pre-recorded Pipeline

```
JSON file (avatar-data-smplx/<label>.smplx.json)
  ↓ (fetch + JSON.parse)
SmplxFrame { label, frameRate, frames: SmplxPose[] }
  ↓ (smplxToAvatarFrames)
AvatarFrame[] (12 bone quaternions per frame)
  ↓ (smoothFramesSpline)
Smoothed AvatarFrame[] (catmull-rom spline via squad)
  ↓ (SignAvatar.tsx animation loop)
Three.js bone.quaternion.slerp() per bone per frame
```

#### 4.2.1 SMPL-X to AvatarFrame Conversion (`smplxConverter.ts`)

**Body joints** (SMPL-X index → Avatar bone):
- Joint 9 (spine_3) → `torso`
- Joint 15 (head) → `head`
- Joint 12 (neck) → `neck`
- Joint 13 (left_clavicle) → `leftShoulder`
- Joint 14 (right_clavicle) → `rightShoulder`
- Joint 16 (left_upper_arm) → `leftUpperArm`
- Joint 17 (right_upper_arm) → `rightUpperArm`
- Joint 18 (left_forearm) → `leftForearm`
- Joint 19 (right_forearm) → `rightForearm`
- Joint 20 (left_hand) → `leftHand` + hand_pose[66:69]
- Joint 21 (right_hand) → `rightHand` + hand_pose[69:72]

**Body pose extraction** (`getBodyJointQuat`):
```typescript
const offset = 3 + (smplxJointIdx - 1) * 3;
const aa = [pose[offset], pose[offset + 1], pose[offset + 2]];
const q = axisAngleToQuat(aa);
const correction = BONE_FRAME_CORRECTION[avatarKey];
return transformQuatByFrame(q, correction);  // R * q * R⁻¹ (conjugation)
```

**Bone Frame Correction** (`BONE_FRAME_CORRECTION`):

| Bone | Correction Quat [x,y,z,w] | Maps From (IK T-pose) | Maps To (Xbot bind) |
|------|--------------------------|----------------------|-------------------|
| leftUpperArm | `[0, 0, 1, 0]` (180° Z) | `[-1, 0, 0]` | `[1, 0, 0]` |
| rightUpperArm | `[0, 0, -1, 0]` (180° Z) | `[1, 0, 0]` | `[-1, 0, 0]` |
| leftForearm | `[0, 0, 0.707, 0.707]` (90° Z) | `[0, -1, 0]` | `[1, 0, 0]` |
| rightForearm | `[0, 0, -0.707, 0.707]` (-90° Z) | `[0, -1, 0]` | `[-1, 0, 0]` |
| leftHand | `[0, 0, 0.707, 0.707]` (90° Z) | `[0, -1, 0]` | `[1, 0, 0]` |
| rightHand | `[0, 0, -0.707, 0.707]` (-90° Z) | `[0, -1, 0]` | `[-1, 0, 0]` |
| neck | `[0.707, 0, 0, 0.707]` (90° X) | `[0, 1, 0]` | `[0, 0, 1]` |
| head | `[0.707, 0, 0, 0.707]` (90° X) | `[0, 1, 0]` | `[0, 0, 1]` |
| torso | `[0, 0, 0, 1]` (identity) | `[0, 1, 0]` | `[0, 1, 0]` |
| shoulders | `[0, 0, 0, 1]` (identity) | dynamic | varies |

**Frame transformation**: `result = R * q * R⁻¹` (quaternion conjugation), which preserves the rotation angle while reorienting the axis from the IK T-pose frame to the Xbot bind-pose frame.

**Hand pose combination** (`smplxConverter.ts:213-236`):
```typescript
// SMPL-X local frame: body_pose hand rotation * hand_pose, then frame correction
const lhBodyQuat = axisAngleToQuat([pose[63], pose[64], pose[65]]);  // joint 20
const leftHandPoseQuat = axisAngleToQuat([pose[66], pose[67], pose[68]]);  // hand_pose
const lhCombined = multiplyQuats(lhBodyQuat, leftHandPoseQuat);  // body_quat * hand_pose
const lhCorrection = BONE_FRAME_CORRECTION["leftHand"];  // [0, 0, 0.707, 0.707]
frame.leftHand = transformQuatByFrame(lhCombined, lhCorrection);  // R * combined * R⁻¹
```

#### 4.2.2 Quaternion-Based Retargeting (Fallback Path)

**Source**: `retarget.ts` → `retargetFrames()` → `smoothFramesSpline()`

**Algorithm** (`retargetFrame`): For each frame:
1. Extract 21 key landmarks (pose, both hands' wrists, index MCPs, pinky MCPs, shoulders, elbows, nose)
2. Compute torso direction: `shoulderMid → nose` (with Z dampening via `dampZ()`)
3. Compute each bone's target direction from landmark-to-landmark vectors
4. Use `alignBoneToDirection()` to compute the quaternion that rotates each bone's default direction to the target direction
5. For bones with roll ambiguity (forearms, hands): use cross-product of bone direction and wrist-to-index/pinky direction as up-vector to disambiguate roll

**Smoothing** (`smoothFramesSpline`):
1. Hemisphere alignment: negate quaternions with negative dot product relative to previous
2. Catmull-Rom spline via Squad (Spherical Quadrangle Interpolation) across 4 frames
3. Uses quaternion logarithmic/exponential maps for proper C1-continuous interpolation

### 4.3 Python SMPL-X Service Pipeline

```
MediaPipe landmarks (345 floats)
  ↓ (SmplxFitter.fit_frame)
1. Y/Z flip: [x, -y, -z]  (MediaPipe → OpenGL)
2. CCDIK solver:
   a. Initialize 21-joint hierarchy in T-pose
   b. Set targets: shoulders, elbows, wrists, nose, hips, ankles, index MCPs
   c. Solve: 15 iterations, tolerance 1e-3
   d. Convert world → local quaternions (local = world * conjugate(parent_world))
3. Extract pose[0:65] body joints as axis-angle
4. Extract hand_pose[66:72] as LOCAL-frame direction vectors
  ↓ (FitResult)
pose: 72 floats, betas: 10, transl: 3, global_orient: 3, smplx_joints: 384, confidence: float
```

#### 4.3.1 CCDIK Solver (`fit_smplx.py:CCDIKChain`)

- **Joint hierarchy**: 21 joints (pelvis → feet → spine → neck/shoulders → arms → hands)
- **T-pose initialization**: Upper arms extend sideways (`[-1,0,0]` for left, `[1,0,0]` for right); forearms/hand point down `[0,-1,0]`
- **CCDIK iterations**: For each joint with a target, process from end-effector toward root:
  - Compute rotation axis = `cross(effector - parent, target - parent)`
  - Compute angle = `arccos(dot(normalized))`
  - Apply rotation to joint and all children
- **World → Local conversion**: Snapshot world quaternions first, then compute `local[i] = world[i] * conjugate(world[parent[i]])`

#### 4.3.2 Hand Pose Target Selection

- **Primary target**: `left_index_mcp` (landmark index 5) and `right_index_mcp` — the direction from wrist to index MCP defines hand orientation
- **Fallback**: If index MCP is collapsed (same as wrist position), fall back to wrist target
- **World → local conversion**: `rotate_vector_by_inverse_quat(world_dir, hand_body_quat)` transforms the world-space direction into the hand's local frame
- **Encoding**: `pose[66:69] = local_dir * 0.1` (scaled to 0.1 magnitude for axis-angle encoding)
- **Shortest-path**: `if quat[3] < 0: quat = -quat` (ensures angle in `[0, π]`)

#### 4.3.3 Batch Conversion

`convert_avatar_data.py` processes all 402 labels:
1. Read `public/avatar-data/<label>.json` (345-float frames)
2. Run `SmplxFitter.fit_frame()` per frame (IK-only mode, ~12s per file)
3. Write `public/avatar-data-smplx/<label>.smplx.json`
4. Update manifest with `frameCount` and `label`

---

## 5. Application and Implementation: Three.js Rendering

### 5.1 Engine Stack

- **Renderer**: `THREE.WebGLRenderer` with antialiasing, PCF soft shadows
- **Composer**: `EffectComposer` → `RenderPass` + `UnrealBloomPass` (strength 0.25, radius 0.3, threshold 0.85) + `OutputPass` (sRGB, ACES Filmic tone mapping)
- **Environment**: `RoomEnvironment` with PMREM for image-based lighting
- **Lighting**: Ambient + directional key light + 4 point lights + 2 colored spotlights

### 5.2 Avatar Model: Xbot GLB

- **File**: `public/models/Xbot.glb`
- **Loader**: `GLTFLoader`
- **Bone mapping** (`BONE_MAP` in `SignAvatar.tsx`):

| GLTF Bone Name | Avatar Key |
|---|---|
| mixamorigHips | root |
| mixamorigSpine | torso |
| mixamorigHead | head |
| mixamorigNeck | neck |
| mixamorigLeftShoulder | leftShoulder |
| mixamorigLeftArm | leftUpperArm |
| mixamorigLeftForeArm | leftForearm |
| mixamorigLeftHand | leftHand |
| mixamorigRightShoulder | rightShoulder |
| mixamorigRightArm | rightUpperArm |
| mixamorigRightForeArm | rightForearm |
| mixamorigRightHand | rightHand |

### 5.3 Bind-Pose Capture

On model load:
1. **Bind pose quaternions** captured for all 11 quaternion bones (`bindPoseQuats`)
2. **Parent world quaternions** captured for 10 bones (`parentWorldQuatsRef`) — used for arm/forearm/hand parent-space alignment
3. These are passed to `loadLabelData()` → `smplxConverter.ts` or `retarget.ts` to transform bone rotations from SMPL-X/MediaPipe space to Xbot local bone space

### 5.4 Per-Frame Animation Application

```typescript
// In the requestAnimationFrame loop:
if (isAnimatingRef.current && targetFrameRef2.current && isPlayingRef.current) {
  const elapsed = t - transitionStartTimeRef.current;
  const duration = transitionDuration / playbackSpeedRef.current;
  const rawT = clamp(elapsed / duration, 0, 1);
  transitionProgressRef.current = easeInOutCubic(rawT);

  for (const key of AVATAR_KEYS) {
    const bone = av.bones[key];
    if (!bone || !cur[key] || !tgt[key]) continue;

    if (isQuatBone) {
      // Skip identity/quaternion-zero rotations
      if (isZero(p1)) { bonesSkipped++; continue; }

      tmpQuat1.set(p1);  // current frame quaternion
      tmpQuat2.set(p2);  // target frame quaternion
      resultQuat.slerpQuaternions(tmpQuat1, tmpQuat2, transitionProgressRef.current);
      resultQuat.normalize();

      const dot = bone.quaternion.dot(resultQuat);
      const angle = Math.acos(Math.min(1, Math.max(-1, Math.abs(dot)))) * 2;
      const damping = angle > MAX_BONE_ANGLE ? SMOOTHING * 0.5 : SMOOTHING;
      bone.quaternion.slerp(resultQuat, damping);
      bone.quaternion.normalize();
    } else {
      // Position bones (root, hips) — though both paths zero these out
      bone.position.lerpVectors(p1, p2, transitionProgressRef.current);
    }
  }
}
```

### 5.5 Idle Animation (Breathing Poses)

When not actively animating (`!isAnimatingRef.current && isPlayingRef.current`):
- Head: sinusoidal rotation around Y and Z (`setFromEuler`)
- Neck: sinusoidal Z rotation
- Shoulders: sinusoidal Z rotation with opposing phases
- Upper arms: sinusoidal X-axis rotation
- Forearms: sinusoidal X-axis rotation with phase offsets
- Hands: sinusoidal X-axis rotation with phase offsets
- Torso: sinusoidal Euler rotation
- Model position Y: `Math.sin(t * 0.6) * 0.008` (subtle bounce)
- Model rotation Z: `Math.sin(t * 0.3) * 0.015 * sway` (subtle sway)

Each idle bone blends at `SMOOTHING = 0.25` toward the computed idle pose combined with its bind-pose quaternion:
```typescript
resultQuat.multiplyQuaternions(idleDelta, bindPoseQuat);
bone.quaternion.slerp(resultQuat, 0.2 * SMOOTHING);
```

### 5.6 Data Source Switching

```typescript
// dataLoader.ts: loadLabelData()
const smplxEnabled = isSmplxServiceAvailable();  // checks window.__useSmplxService

if (smplxEnabled) {
  const smplxData = await loadLabelDataSmplx(label, parentWorldQuats);
  if (smplxData) {
    dataCache.set(label, smplxData);  // Cache and use
    return smplxData;
  }
  // Falls through to quaternion path if SMPL-X data unavailable
}

// Fallback: quaternion retargeting
const data: AvatarData = await fetch(`/avatar-data/${sourceFile}`).json();
const rawFrames = retargetFrames(dataArray, parentWorldQuats);
const smoothed = smoothFramesSpline(rawFrames, 5);
```

### 5.7 Live Detection Integration

```typescript
// App.tsx: useHolisticPipeline callback
const { ...pipelineState } = useHolisticPipeline((result: LandmarksResult) => {
  const normalized = normalizeLandmarks(result);
  const features = selectFeatures(normalized.vector);
  inference.pushFrame(features);  // Feeds SlidingWindowBuffer

  if (prediction.label && !currentSign) {
    // Trigger playback of predicted sign
    setSignSequenceState([prediction.label]);
  }
});
```

---

## 6. Data Volumes

| Asset | Format | Size | Count | Total |
|-------|--------|------|-------|-------|
| Pose data files | JSON (345 floats/frame) | ~1.3KB/frame | 402 files | ~260 MB |
| SMPL-X data files | JSON (72+384 floats/frame) | ~7KB/frame | 402 files | ~1.57 GB |
| Avatar model | GLB (mesh + skeleton) | ~7 MB | 1 | ~7 MB |
| TFJS model | JSON + binary weights | ~2 MB | 1 | ~2 MB |
| MediaPipe WASM | WebAssembly | ~10 MB | 1 | ~10 MB |

**Frame counts**: 193,152 total frames across 402 signs (avg ~480 frames/sign)

---

## 7. Key Algorithms Summary

| Algorithm | Location | Purpose |
|-----------|----------|---------|
| One Euro Filter | `smoothing.ts` | Real-time landmark jitter reduction |
| Quaternion SLERP | `retarget.ts:272` | Smooth interpolation between bone rotations |
| Squad (C1 spline) | `retarget.ts:305` | Catmull-Rom-like quaternion curve smoothing |
| CCDIK | `fit_smplx.py:237` | Inverse kinematics for SMPL-X joint fitting |
| World→Local quat | `fit_smplx.py:365` | Convert world-space IK rotations to SMPL-X local rotations |
| Frame conjugation | `smplxConverter.ts:110` | Transform IK bone directions to Xbot bind-pose frame |
| Shortest-arc align | `retarget.ts:124` | Align bone to target direction with up-vector twist constraint |
| MediaPipe→GLTF flip | `retarget.ts:59` | Coordinate system conversion: [x, -y, -z] |
