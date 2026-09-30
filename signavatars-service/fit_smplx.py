"""
SMPL-X fitting logic: converts MediaPipe Holistic landmarks to SMPL-X pose parameters.

MediaPipe Holistic landmark mapping:
  Pose (33 landmarks): 0=nose, 11=L-shoulder, 12=R-shoulder, 13=L-elbow, 14=R-elbow,
    15=L-wrist, 16=R-wrist, 17-20=finger tips, 21-22=thumbs, 23=L-hip, 24=R-hip,
    25=L-knee, 26=R-knee, 27=L-ankle, 28=R-ankle, 29-32=heel/foot
  Left hand (21): 0=wrist, 4=thumb tip, 8=index tip, 12=middle tip, 16=pinky MCP, 20=pinky tip
  Right hand (21): same layout
  Face (40): subset of facial landmarks at indices 225..344

SMPL-X uses 10 shape betas, 3 global orient, 63 body pose (21 joints x 3),
60 hand poses (2 hands x 15), and 9 facial expression params.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# MediaPipe-to-SMPL-X correspondence table
# ---------------------------------------------------------------------------
#
# Each entry maps a MediaPipe landmark index to a SMPL-X joint name.
# SMPL-X joint names follow the convention used in the `smplx` library.
#
# Pose landmarks (offsets 0..98 in the packed buffer, indices 0..32)
POSE_LANDMARK_NAMES = {
    0:  "head_top",          # nose → head_top proxy
    1:  "left_eye_inner",
    2:  "left_eye",
    3:  "left_eye_outer",
    4:  "right_eye_inner",
    5:  "right_eye",
    6:  "right_eye_outer",
    7:  "left_ear",
    8:  "right_ear",
    9:  "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky_finger",
    18: "right_pinky_finger",
    19: "left_index_finger",
    20: "right_index_finger",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

HAND_LANDMARK_OFFSETS = {
    0:  "wrist",
    1:  "thumb_cmc",
    2:  "thumb_mcp",
    3:  "thumb_ip",
    4:  "thumb_tip",
    5:  "index_mcp",
    6:  "index_pip",
    7:  "index_dip",
    8:  "index_tip",
    9:  "middle_mcp",
    10: "middle_pip",
    11: "middle_dip",
    12: "middle_tip",
    13: "ring_mcp",
    14: "ring_pip",
    15: "ring_dip",
    16: "ring_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip",
}

# Packed buffer offsets (must match src/mediapipe/types.ts)
POSE_OFFSET = 0
POSE_COUNT = 33
LEFT_HAND_OFFSET = POSE_COUNT * 3       # 99
HAND_COUNT = 21
RIGHT_HAND_OFFSET = (POSE_COUNT + HAND_COUNT) * 3  # 162
# 345 floats total per frame = 115 landmarks × 3
# 33 pose + 21 left hand + 21 right hand = 75 landmarks (225 floats)
# Remaining 120 floats = 40 face landmarks at offset 225

FACE_START_OFFSET = RIGHT_HAND_OFFSET + HAND_COUNT * 3  # 225
FACE_LANDMARK_COUNT = 40  # 345 - 225 = 120 floats = 40 landmarks

TOTAL_FLOATS_PER_FRAME = 345


def extract_landmarks(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Extract normalized 3D landmarks from a flat 345-float frame into named arrays."""
    if frame.shape == (345,):
        frame = frame.reshape(115, 3)
    elif frame.shape == (115, 3):
        pass
    else:
        raise ValueError(f"Unexpected landmark shape: {frame.shape}")

    result: dict[str, np.ndarray] = {}

    # Pose landmarks (indices 0..32)
    for mp_idx, name in POSE_LANDMARK_NAMES.items():
        result[name] = frame[mp_idx].copy()

    # Left hand landmarks (indices 33..53 → buffer 99..161)
    left_base = POSE_COUNT  # 33
    for h_idx, suffix in HAND_LANDMARK_OFFSETS.items():
        result[f"left_{suffix}"] = frame[left_base + h_idx].copy()

    # Right hand landmarks (indices 54..74 → buffer 162..224)
    right_base = POSE_COUNT + HAND_COUNT  # 54
    for h_idx, suffix in HAND_LANDMARK_OFFSETS.items():
        result[f"right_{suffix}"] = frame[right_base + h_idx].copy()

    # Face landmarks (indices 75..114 → buffer 225..344)
    face_base = right_base + HAND_COUNT  # 75
    for i in range(FACE_LANDMARK_COUNT):
        result[f"face_{i}"] = frame[face_base + i].copy()

    return result


# ---------------------------------------------------------------------------
# SMPL-X joint definitions (simplified subset for upper-body fitting)
# ---------------------------------------------------------------------------

SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine_1",            # 0-3
    "left_knee", "right_knee", "spine_2",                     # 4-6
    "left_ankle", "right_ankle", "spine_3",                   # 7-9
    "left_foot_side", "right_foot_side", "neck",              # 10-12
    "left_clavicle", "right_clavicle", "head",                # 13-15
    "left_upper_arm", "right_upper_arm",                      # 16-17
    "left_elbow", "right_elbow",                             # 18-19
    "left_forearm", "right_forearm",                          # 20-21
    "left_hand", "right_hand",                                # 22-23
    # Hand joints (SMPL-X has 15 per hand = 30 total, indices 25-54)
    "left_thumb_1", "left_thumb_2", "left_thumb_3",           # 24-26
    "left_index_1", "left_index_2", "left_index_3",            # 27-29
    "left_middle_1", "left_middle_2", "left_middle_3",        # 30-32
    "left_ring_1", "left_ring_2", "left_ring_3",              # 33-35
    "left_pinky_1", "left_pinky_2", "left_pinky_3",            # 36-38
    "right_thumb_1", "right_thumb_2", "right_thumb_3",         # 39-41
    "right_index_1", "right_index_2", "right_index_3",          # 42-44
    "right_middle_1", "right_middle_2", "right_middle_3",       # 45-47
    "right_ring_1", "right_ring_2", "right_ring_3",            # 48-50
    "right_pinky_1", "right_pinky_2", "right_pinky_3",          # 51-53
    # Face (expression-driven, indices 54+)
    "jaw", "left_eye", "right_eye",                            # 54-56
    "left_eyebrow_1", "left_eyebrow_2", "left_eyebrow_3",      # 57-59
    "right_eyebrow_1", "right_eyebrow_2", "right_eyebrow_3",   # 60-62
    "mouth_1", "mouth_2", "mouth_3", "mouth_4",                # 63-66
    "mouth_5", "mouth_6", "mouth_7", "mouth_8",                # 67-70
    "mouth_9", "mouth_10", "mouth_11", "mouth_12",            # 71-74
    "mouth_13", "mouth_14", "mouth_15", "mouth_16",           # 75-78
    "left_nostril", "right_nostril",                           # 79-80
    "left_eye_lid", "right_eye_lid",                           # 81-82
    # Total: 83 joints + 3 head_top/face points = 86
]

# Mapping from landmark-derived names to SMPL-X body joint names
# (used to build the correspondence for optimization)
MP_TO_SMPLX_MAP: dict[str, str] = {
    "nose": "head_top",
    "left_shoulder": "left_shoulder",      # → left_clavicle in SMPL-X
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_hand",
    "right_wrist": "right_hand",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_heel": "left_heel",
    "right_heel": "right_heel",
    "left_foot_index": "left_foot",
    "right_foot_index": "right_foot",
    "left_index_tip": "left_index_tip",
    "right_index_tip": "right_index_tip",
    "left_pinky_finger": "left_pinky_tip",
    "right_pinky_finger": "right_pinky_tip",
    "left_thumb": "left_thumb_tip",
    "right_thumb": "right_thumb_tip",
    # Hand wrist points
    "left_wrist": "left_hand",
    "right_wrist": "right_hand",
}


# ---------------------------------------------------------------------------
# Axis-angle utilities
# ---------------------------------------------------------------------------

def axis_angle_to_rotvec(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3 values) to rotation vector (same representation).
    SMPL-X uses axis-angle where the vector direction is the axis and magnitude is the angle."""
    return np.asarray(aa, dtype=np.float64)


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to quaternion [x, y, z, w]."""
    theta = np.linalg.norm(rotvec)
    if theta < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / theta
    half = theta / 2.0
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])


# ---------------------------------------------------------------------------
# IK solver: place chain of joints using CCDIK
# ---------------------------------------------------------------------------

class CCDIKChain:
    """Simple Cyclic-Coordinate-Descent IK for a kinematic chain of joints.
    
    Computes local joint rotations (relative to parent) as the rotation from
    each joint's initial bone direction to its final solved bone direction.
    """

    def __init__(self, joint_names: list[str], joint_parents: list[int | None]):
        self.joint_names = joint_names
        self.joint_parents = joint_parents
        self.joint_positions = np.zeros((len(joint_names), 3))
        self.joint_rotations: list[np.ndarray] = [
            np.array([0.0, 0.0, 0.0, 1.0]) for _ in joint_names
        ]
        self._initial_dirs: list[np.ndarray | None] = [None] * len(joint_names)

    def _capture_initial_directions(self):
        """Capture initial bone directions (from parent to joint) before solving."""
        for i in range(len(self.joint_names)):
            parent_idx = self.joint_parents[i]
            if parent_idx is not None:
                direction = self.joint_positions[i] - self.joint_positions[parent_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    self._initial_dirs[i] = direction / norm
                else:
                    self._initial_dirs[i] = None
            else:
                self._initial_dirs[i] = None

    def solve(self, targets: dict[str, np.ndarray], iterations: int = 20, tolerance: float = 1e-4):
        """Solve IK using Cyclic Coordinate Descent.
        
        For each joint that has a target, standard CCDIK processes each joint
        from end-effector toward root, rotating the joint to bring the
        TARGETED joint closer to its target. After all iterations, computes
        final joint rotations as the delta from initial to final bone direction.
        """
        self._capture_initial_directions()
        
        for _ in range(iterations):
            solved = True
            for joint_idx in range(len(self.joint_names) - 1, -1, -1):
                name = self.joint_names[joint_idx]
                if name not in targets:
                    continue

                target = targets[name]
                effector = self.joint_positions[joint_idx]
                error = np.linalg.norm(target - effector)
                if error < tolerance:
                    continue

                solved = False

                parent_idx = self.joint_parents[joint_idx]
                if parent_idx is None:
                    self.joint_positions[joint_idx] += (target - effector) * 0.3
                    continue

                parent_pos = self.joint_positions[parent_idx]

                to_eff = effector - parent_pos
                to_target = target - parent_pos

                if np.linalg.norm(to_eff) < 1e-10 or np.linalg.norm(to_target) < 1e-10:
                    continue

                axis = np.cross(to_eff, to_target)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-10:
                    continue
                axis = axis / axis_norm
                angle = np.arccos(np.clip(np.dot(to_eff, to_target) /
                                          (np.linalg.norm(to_eff) * np.linalg.norm(to_target)), -1, 1))

                # Apply rotation to this joint AND all children
                rot = R.from_rotvec(axis * angle)
                rotation_matrix = rot.as_matrix()

                for idx in [joint_idx] + self._get_children(joint_idx):
                    rel_pos = self.joint_positions[idx] - parent_pos
                    self.joint_positions[idx] = parent_pos + rotation_matrix @ rel_pos

            if solved:
                break

        # Compute final joint rotations from initial to final bone directions
        for i in range(len(self.joint_names)):
            parent_idx = self.joint_parents[i]
            if parent_idx is None:
                continue
            if self._initial_dirs[i] is None:
                continue
            
            initial_dir = self._initial_dirs[i]
            current_dir = self.joint_positions[i] - self.joint_positions[parent_idx]
            current_norm = np.linalg.norm(current_dir)
            if current_norm < 1e-10:
                continue
            current_dir = current_dir / current_norm

            # Compute rotation that aligns initial_dir to current_dir
            dot = np.dot(initial_dir, current_dir)
            if dot > 0.99999:
                joint_rot = R.from_rotvec(np.array([0.0, 0.0, 0.0]))
            elif dot < -0.99999:
                # 180 degree rotation
                ortho = np.array([1.0, 0, 0]) if abs(initial_dir[0]) < 0.9 else np.array([0, 1.0, 0])
                rot_axis = np.cross(initial_dir, ortho)
                rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-10)
                joint_rot = R.from_rotvec(rot_axis * np.pi)
            else:
                rot_axis = np.cross(initial_dir, current_dir)
                rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-10)
                rot_angle = np.arccos(np.clip(dot, -1, 1))
                joint_rot = R.from_rotvec(rot_axis * rot_angle)
            
            q = joint_rot.as_quat(scalar_first=False)  # [x, y, z, w]
            self.joint_rotations[i] = np.array([q[0], q[1], q[2], q[3]])

        # Convert world-space rotations to local-space rotations.
        # The rotation above takes initial_dir (world) to current_dir (world),
        # which is a WORLD-SPACE rotation. SMPL-X body_pose expects LOCAL rotations
        # (relative to parent). The chain relation is
        #   R(world_child) = R(parent_world) . R(local_child)
        # so inverting it gives
        #   R(local_child) = R(parent_world)^-1 . R(world_child)
        # i.e. local = conjugate(parent_world) * world.
        # We must use the PARENT'S WORLD rotation (not its local), so we snapshot
        # world rotations before overwriting with local rotations.
        #
        # E7 FIX (PHASE 6): the operand order was reversed here. It read
        #   local = _multiply_quats(world_q, parent_inv)   # R(world) . R(parent)^-1
        # Quaternion products do not commute, so that is NOT the inverse of the
        # chain relation; it only happens to be right when the parent is
        # unrotated, which is why the root looked fine and the error grew with
        # chain depth (measured 0.00deg -> 5.78deg -> 10.80deg over three
        # joints, i.e. twisted limbs). Proven by
        # diagnose_e7_composition.py, which round-trips the conversion.
        world_rotations = [q.copy() for q in self.joint_rotations]

        for i in range(len(self.joint_names)):
            parent_idx = self.joint_parents[i]
            if parent_idx is None:
                # Root: local == world (no parent)
                continue
            parent_world = world_rotations[parent_idx]
            parent_inv = np.array([-parent_world[0], -parent_world[1], -parent_world[2], parent_world[3]])
            world_q = world_rotations[i]
            # local = conjugate(parent_world) * world   (order matters!)
            self.joint_rotations[i] = self._multiply_quats(parent_inv, world_q)

    @staticmethod
    def _multiply_quats(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply quaternions a * b (x, y, z, w format). Result = apply b first, then a."""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    def _get_children(self, idx: int) -> list[int]:
        children = []
        for i, parent in enumerate(self.joint_parents):
            if parent == idx:
                children.append(i)
        return children


def rotate_vector_by_inverse_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q (x, y, z, w format).

    This transforms a world-space vector into the local frame defined by q.
    Equivalent to: conjugate(q) * [0, v] * q
    """
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # Build the rotation matrix for the inverse quaternion
    # (same as forward rotation with -qx, -qy, -qz)
    rx, ry, rz, rw = -qx, -qy, -qz, qw
    xx, yy, zz = rx * rx, ry * ry, rz * rz
    xy, xz, yz = rx * ry, rx * rz, ry * rz
    xw, yw, zw = rx * rw, ry * rw, rz * rw
    return np.array([
        vx * (1 - 2 * (yy + zz)) + vy * (2 * (xy - zw)) + vz * (2 * (xz + yw)),
        vx * (2 * (xy + zw)) + vy * (1 - 2 * (xx + zz)) + vz * (2 * (yz - xw)),
        vx * (2 * (xz - yw)) + vy * (2 * (yz + xw)) + vz * (1 - 2 * (xx + yy)),
    ])

@dataclass
class FitResult:
    pose: list[float]           # 72 floats (23 joints × 3 axis-angle)
    betas: list[float]          # 10 floats (shape parameters)
    transl: list[float]         # 3 floats (root translation)
    global_orient: list[float]  # 3 floats (global rotation)
    smplx_joints: list[float]   # 128 × 3 = 384 floats (joint positions)
    confidence: float


class SmplxFitter:
    """Fits SMPL-X pose parameters from MediaPipe Holistic landmarks.

    Two strategies are supported:
    1. Full SMPL-X model fitting (requires smplx library + model files)
    2. IK-based fitting (fallback, no model files needed)

    The IK fallback produces pose parameters by solving inverse kinematics
    for each limb chain using CCDIK, which works without downloading the
    SMPL-X model. This maintains parity with the TypeScript retargeting
    while providing the SMPL-X parameter format for downstream consumption.
    """

    # SMPL-X body joint hierarchy (subset for fitting)
    # Indices match the SMPL-X body pose parameter ordering:
    #   0: root, 1: left_hip, 2: right_hip, 3: spine_1,
    #   4: left_knee, 5: right_knee, 6: spine_2,
    #   7: left_ankle, 8: right_ankle, 9: spine_3,
    #   10: left_foot_side, 11: right_foot_side, 12: neck,
    #   13: left_clavicle, 14: right_clavicle, 15: head,
    #   16: left_upper_arm, 17: right_upper_arm,
    #   18: left_elbow (not a separate SMPL-X param), 19: right_elbow
    #   20: left_forearm (not a separate SMPL-X param), 21: right_forearm
    #   22: left_hand, 23: right_hand
    # Note: SMPL-X body_pose has 63 values = 21 joints × 3 (excluding root)
    # Total pose = 3 (global_orient) + 63 (body_pose) + 90 (hand_pose) + 12 (expr) = 168
    # But commonly used: 72 = 3 (global_orient) + 63 (body_pose) + 6 (hand_pose for both)
    # We'll use the standard 72-dim pose (global_orient + body_pose + 6D hand pose minimal)
    BODY_JOINT_HIERARCHY = {
        "pelvis":      {"parent": None,           "smpl_idx": 0},
        "left_hip":    {"parent": "pelvis",       "smpl_idx": 1},
        "right_hip":   {"parent": "pelvis",       "smpl_idx": 2},
        "spine_1":     {"parent": "pelvis",       "smpl_idx": 3},
        "left_knee":   {"parent": "left_hip",     "smpl_idx": 4},
        "right_knee":  {"parent": "right_hip",    "smpl_idx": 5},
        "spine_2":     {"parent": "spine_1",      "smpl_idx": 6},
        "left_ankle":  {"parent": "left_knee",    "smpl_idx": 7},
        "right_ankle": {"parent": "right_knee",   "smpl_idx": 8},
        "spine_3":     {"parent": "spine_2",      "smpl_idx": 9},
        "neck":        {"parent": "spine_3",      "smpl_idx": 12},
        "left_clavicle":  {"parent": "spine_3",  "smpl_idx": 13},
        "right_clavicle": {"parent": "spine_3",  "smpl_idx": 14},
        "head":        {"parent": "neck",         "smpl_idx": 15},
        "left_upper_arm":  {"parent": "left_clavicle",  "smpl_idx": 16},
        "right_upper_arm": {"parent": "right_clavicle", "smpl_idx": 17},
        "left_forearm":    {"parent": "left_upper_arm",  "smpl_idx": 18},
        "right_forearm":   {"parent": "right_upper_arm", "smpl_idx": 19},
        "left_hand":       {"parent": "left_forearm",    "smpl_idx": 20},
        "right_hand":      {"parent": "right_forearm",   "smpl_idx": 21},
    }

    # MediaPipe landmark indices for body joints
    MP_BODY_LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    def __init__(self, model_dir: str | None = None, gender: str = "NEUTRAL"):
        self.model_dir = model_dir
        self.gender = gender
        self._smplx_model = None
        self._use_full_model = False

        # Try to load full SMPL-X model
        try:
            import smplx
            if model_dir and os.path.isdir(model_dir):
                model_path = os.path.join(model_dir, f"SMPLX_{gender}_v1.1", "smplx")
                if os.path.isdir(model_path) or os.path.isdir(model_dir):
                    self._smplx_model = smplx.create(
                        model_dir, model_type="smplx", gender=gender.lower(),
                        use_face_contour=True, num_comp=12,
                    )
                    self._use_full_model = True
                    print(f"[fitter] Loaded SMPL-X model from {model_dir}")
        except Exception as e:
            print(f"[fitter] SMPL-X model not available, falling back to IK mode: {e}")
            self._use_full_model = False

    def fit_frame(
        self,
        landmarks: np.ndarray,
        parent_world_quats: dict[str, list[float]] | None = None,
    ) -> FitResult:
        """Fit SMPL-X pose parameters for a single frame.

        Args:
            landmarks: 345 floats (115 landmarks × 3), already normalized to
                shoulder midpoint with shoulder-width scale.
            parent_world_quats: Optional parent bone world quaternions (from the
                bind-pose of the Xbot model). Used to map SMPL-X poses to the
                Three.js bone hierarchy.

        Returns:
            FitResult with pose, betas, transl, global_orient, smplx_joints, confidence.
        """
        if landmarks.shape == (345,):
            landmarks_3d = landmarks.reshape(115, 3)
        elif landmarks.shape == (115, 3):
            landmarks_3d = landmarks
        else:
            raise ValueError(f"Expected 345 or (115,3) shape, got {landmarks.shape}")

        # MediaPipe Y is down, Z is inward. Convert to OpenGL-style: Y up, Z out.
        # The landmarks passed in have already been normalized to shoulder midpoint.
        mp_lm = extract_landmarks(landmarks_3d)

        # MediaPipe to OpenGL conversion
        for name in list(mp_lm.keys()):
            v = mp_lm[name]
            mp_lm[name] = np.array([v[0], -v[1], -v[2]])

        if self._use_full_model:
            result = self._fit_with_smplx(mp_lm)
        else:
            result = self._fit_with_ik(mp_lm)

        return result

    def _fit_with_smplx(self, mp_lm: dict[str, np.ndarray]) -> FitResult:
        """Full SMPL-X fitting using the smplx library."""
        import torch

        # Build target joint positions from MediaPipe landmarks
        target_joints = np.zeros((len(SMPLX_JOINT_NAMES), 3))

        # Map landmarks to SMPL-X joints
        for mp_name, smpl_name in MP_TO_SMPLX_MAP.items():
            if mp_name in mp_lm and smpl_name in SMPLX_JOINT_NAMES:
                idx = SMPLX_JOINT_NAMES.index(smpl_name)
                target_joints[idx] = mp_lm[mp_name]

        # Use SMPLify-XL for pose optimization
        # This is a simplified version - full implementation would use
        # the smplx library's optimization routines

        body_pose = np.zeros(63)  # 21 joints × 3
        left_hand_pose = np.zeros(15 * 3)  # 15 joints × 3
        right_hand_pose = np.zeros(15 * 3)
        betas = np.zeros(10)
        transl = np.zeros(3)
        global_orient = np.zeros(3)

        # Simple assignment based on direct correspondence
        # In production, this would be replaced with gradient-based optimization
        for mp_name, smpl_name in MP_TO_SMPLX_MAP.items():
            if mp_name in mp_lm and smpl_name in self.BODY_JOINT_HIERARCHY:
                info = self.BODY_JOINT_HIERARCHY[smpl_name]
                smpl_idx = info["smpl_idx"]
                if smpl_idx < 72 // 3:  # Only fit body joints
                    pose_idx = smpl_idx * 3
                    if pose_idx < 63:
                        # This would be where gradient-based fitting happens
                        # For now, leave as init (zeros = T-pose)
                        pass

        # Assemble SMPL-X parameters
        # pose = global_orient (3) + body_pose (63) + hand_pose (90)
        # But we use the reduced 72-dim format for the converter
        full_pose = np.concatenate([global_orient, body_pose, np.zeros(6)])  # 72 total

        # Forward pass to get joint positions
        if self._smplx_model is not None:
            import torch
            with torch.no_grad():
                out = self._smplx_model(
                    global_orient=torch.tensor([global_orient], dtype=torch.float32),
                    body_pose=torch.tensor([body_pose.reshape(1, -1)], dtype=torch.float32),
                    betas=torch.tensor([betas], dtype=torch.float32),
                    transl=torch.tensor([transl], dtype=torch.float32),
                    left_hand_pose=torch.tensor([left_hand_pose.reshape(1, -1)], dtype=torch.float32),
                    right_hand_pose=torch.tensor([right_hand_pose.reshape(1, -1)], dtype=torch.float32),
                    return_verts=True,
                )
                joints = out.joints.squeeze(0).numpy()
                # SMPL-X returns 127 joints
                smplx_joints_flat = joints.flatten().tolist()
                confidence = 0.8  # moderate confidence without optimization
        else:
            smplx_joints_flat = [0.0] * (128 * 3)
            confidence = 0.0

        return FitResult(
            pose=full_pose.tolist(),
            betas=betas.tolist(),
            transl=transl.tolist(),
            global_orient=global_orient.tolist(),
            smplx_joints=smplx_joints_flat,
            confidence=confidence,
        )

    def _fit_with_ik(self, mp_lm: dict[str, np.ndarray]) -> FitResult:
        """IK-based fitting: solve for joint rotations using CCDIK.

        This is the fallback when the SMPL-X model is not available.
        It produces pose parameters that match the MediaPipe landmark positions
        by solving inverse kinematics for each limb chain.
        """
        # Build joint hierarchy for IK
        #
        # PHASE 13 FIX -- INDEX MAPPING BUG.
        #
        # `smplx_joints` below is written POSITIONALLY into a 128x3 array and is
        # read back by src/avatar/smplxAdapter.ts through IK_BODY_IDX. Both sides
        # must agree on the index of every joint or the wrong joint is read.
        #
        # The bug: this list had 20 names but omitted four joints that exist in
        # the official SMPL-X layout:
        #     left_foot_side / right_foot_side   (official 10, 11)
        #     left_elbow / right_elbow           (official 18, 19)
        # so every index from 10 onward was shifted by 4. The reader resolved
        # the neck as a foot, the clavicles as neck/head, and -- worst -- read
        # the WRIST from slot 16, which this list never filled, so the wrist
        # position was ZERO. Every arm was then aimed from a zero-length vector
        # and both arms folded into the torso.
        #
        # IK_BODY_IDX in smplxAdapter.ts is FROZEN by this phase (no src/avatar
        # edits allowed), so the writer must emit exactly the order the reader
        # expects: the 20-entry IK table. The fix is therefore to name the
        # entries for the POSITION each slot holds -- "left_forearm" is the WRIST
        # slot, "left_hand" the hand centre -- rather than for the bone that
        # precedes it, and to make sure every one of the 20 slots is written.
        #
        # Note this matches the reader's documented convention: "the joint at
        # index i represents the END of bone i".
        joint_names_ordered = [
            "pelvis", "left_hip", "right_hip", "spine_1",
            "left_knee", "right_knee", "spine_2",
            "left_ankle", "right_ankle", "spine_3",
            "neck", "left_clavicle", "right_clavicle", "head",
            # 14,15 = elbow positions (end of the upper-arm bone)
            "left_elbow", "right_elbow",
            # 16,17 = WRIST positions (end of the forearm bone)
            "left_forearm", "right_forearm",
            # 18,19 = hand centres
            "left_hand", "right_hand",
        ]

        # Parent indices (None for root)
        name_to_idx = {name: i for i, name in enumerate(joint_names_ordered)}
        parent_names = {
            "pelvis": None,
            "left_hip": "pelvis", "right_hip": "pelvis", "spine_1": "pelvis",
            "left_knee": "left_hip", "right_knee": "right_hip", "spine_2": "spine_1",
            "left_ankle": "left_knee", "right_ankle": "right_knee", "spine_3": "spine_2",
            "neck": "spine_3", "left_clavicle": "spine_3", "right_clavicle": "spine_3",
            "head": "neck",
            "left_elbow": "left_clavicle", "right_elbow": "right_clavicle",
            "left_forearm": "left_elbow", "right_forearm": "right_elbow",
            "left_hand": "left_forearm", "right_hand": "right_forearm",
        }

        parents = [name_to_idx.get(parent_names[name]) if parent_names[name] else None
                   for name in joint_names_ordered]

        # Initialize joint positions from landmarks
        ik = CCDIKChain(joint_names_ordered, parents)

        # Extract landmark-derived positions
        l_hip = mp_lm.get("left_hip", np.zeros(3))
        r_hip = mp_lm.get("right_hip", np.zeros(3))
        l_shoulder = mp_lm.get("left_shoulder", np.zeros(3))
        r_shoulder = mp_lm.get("right_shoulder", np.zeros(3))
        l_elbow = mp_lm.get("left_elbow", np.zeros(3))
        r_elbow = mp_lm.get("right_elbow", np.zeros(3))
        l_wrist = mp_lm.get("left_wrist", np.zeros(3))
        r_wrist = mp_lm.get("right_wrist", np.zeros(3))
        l_knee = mp_lm.get("left_knee", np.zeros(3))
        r_knee = mp_lm.get("right_knee", np.zeros(3))
        l_ankle = mp_lm.get("left_ankle", np.zeros(3))
        r_ankle = mp_lm.get("right_ankle", np.zeros(3))
        nose_pos = mp_lm.get("nose", np.zeros(3))

        # Estimate body proportions from landmarks
        hip_center = (l_hip + r_hip) / 2
        shoulder_center = (l_shoulder + r_shoulder) / 2
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        torso_height = np.linalg.norm(shoulder_center - hip_center)

        # Arm lengths
        l_upperarm_len = np.linalg.norm(l_elbow - l_shoulder)
        l_forearm_len = np.linalg.norm(l_wrist - l_elbow)
        r_upperarm_len = np.linalg.norm(r_elbow - r_shoulder)
        r_forearm_len = np.linalg.norm(r_wrist - r_elbow)

        # --- T-Pose initialization ---
        # Root (pelvis) at hip center
        ik.joint_positions[name_to_idx["pelvis"]] = hip_center.copy()

        # Hips (spread slightly in T-pose)
        hip_offset = shoulder_width * 0.12
        ik.joint_positions[name_to_idx["left_hip"]] = hip_center + np.array([-hip_offset, 0, 0])
        ik.joint_positions[name_to_idx["right_hip"]] = hip_center + np.array([hip_offset, 0, 0])

        # Knees (below hips, estimate from hip to ankle distance)
        l_leg_len = np.linalg.norm(l_ankle - l_hip) if np.linalg.norm(l_ankle - l_hip) > 0.01 else 0.4
        r_leg_len = np.linalg.norm(r_ankle - r_hip) if np.linalg.norm(r_ankle - r_hip) > 0.01 else 0.4
        ik.joint_positions[name_to_idx["left_knee"]] = hip_center + np.array([-hip_offset, -l_leg_len * 0.5, 0])
        ik.joint_positions[name_to_idx["right_knee"]] = hip_center + np.array([hip_offset, -r_leg_len * 0.5, 0])

        # Ankles (at landmark positions)
        ik.joint_positions[name_to_idx["left_ankle"]] = l_ankle.copy()
        ik.joint_positions[name_to_idx["right_ankle"]] = r_ankle.copy()

        # Spine chain stacked vertically from pelvis
        if torso_height > 0.01:
            spine_step = torso_height / 3
        else:
            spine_step = 0.15
        spine1_pos = hip_center + np.array([0, spine_step, 0])
        spine2_pos = spine1_pos + np.array([0, spine_step, 0])
        spine3_pos = spine2_pos + np.array([0, spine_step, 0])
        ik.joint_positions[name_to_idx["spine_1"]] = spine1_pos
        ik.joint_positions[name_to_idx["spine_2"]] = spine2_pos
        ik.joint_positions[name_to_idx["spine_3"]] = spine3_pos

        # Neck above spine3
        neck_pos = spine3_pos + np.array([0, 0.1, 0])
        ik.joint_positions[name_to_idx["neck"]] = neck_pos

        # Head — start at T-pose (above neck) so IK rotates toward nose
        head_init = neck_pos + np.array([0, 0.15, 0])
        ik.joint_positions[name_to_idx["head"]] = head_init

        # Clavicles — positioned at shoulder landmarks (T-pose: clavicle runs horizontally to shoulders)
        ik.joint_positions[name_to_idx["left_clavicle"]] = l_shoulder.copy()
        ik.joint_positions[name_to_idx["right_clavicle"]] = r_shoulder.copy()

        # Upper arms in T-pose (arms out to the sides, matching SMPL-X zero pose).
        # PHASE 13: slot 14/15 is the ELBOW (end of the upper-arm bone) and slot
        # 16/17 is the WRIST (end of the forearm bone). Both used to be written
        # under the "upper_arm" name, which meant the wrist slot the reader reads
        # was never filled and stayed zero.
        l_arm_dir = np.array([-1, 0, 0])
        r_arm_dir = np.array([1, 0, 0])
        l_elbow_pos = l_shoulder + l_arm_dir * l_upperarm_len
        r_elbow_pos = r_shoulder + r_arm_dir * r_upperarm_len
        ik.joint_positions[name_to_idx["left_elbow"]] = l_elbow_pos
        ik.joint_positions[name_to_idx["right_elbow"]] = r_elbow_pos

        # Forearm (WRIST slot) in T-pose: from the elbow, hanging down.
        # PHASE 13: the wrist now hangs off "left_elbow" (slot 14) rather than
        # the removed "left_upper_arm" name, so this slot is actually written.
        l_forearm_dir = np.array([0, -1, 0])
        r_forearm_dir = np.array([0, -1, 0])
        ik.joint_positions[name_to_idx["left_forearm"]] = l_elbow_pos + l_forearm_dir * l_forearm_len
        ik.joint_positions[name_to_idx["right_forearm"]] = r_elbow_pos + r_forearm_dir * r_forearm_len

        # Hands at end of forearm chains (hand centre in T-pose)
        ik.joint_positions[name_to_idx["left_hand"]] = ik.joint_positions[name_to_idx["left_forearm"]] + l_forearm_dir * l_forearm_len
        ik.joint_positions[name_to_idx["right_hand"]] = ik.joint_positions[name_to_idx["right_forearm"]] + r_forearm_dir * r_forearm_len

        # Target positions: actual landmark positions
        # In the IK hierarchy, each joint position = end of that bone (toward child)
        # - left_upper_arm joint position = elbow position → target left_elbow
        # - left_forearm joint position = wrist position -> target left_wrist
        # - left_hand joint position = hand center -> target left_wrist
        # Key fixes:
        # - neck targets nose (head_top) to orient forward head tilt
        # - clavicles target shoulder landmarks (constrains shoulder tilt)
        # - head is NOT targeted (follows from neck)
        # - spine_3 is NOT targeted (torso stays upright — natural for signing)
        IK_TO_LANDMARK = {
            "pelvis": None,
            "left_hip": None,
            "right_hip": None,
            "spine_1": None,
            "left_knee": None,
            "right_knee": None,
            "spine_2": None,
            "left_ankle": None,
            "right_ankle": None,
            "spine_3": None,  # passive - torso stays upright (natural for signing)
            "neck": "head_top",  # nose position
            "head": None,  # follows from neck
            "left_clavicle": "left_shoulder",  # constrain shoulder position
            "right_clavicle": "right_shoulder",
            "left_elbow": "left_elbow",  # constrain elbow position (end of upper arm)
            "right_elbow": "right_elbow",
            "left_forearm": "left_wrist",     # constrain wrist position
            "right_forearm": "right_wrist",
            "left_hand": "left_index_mcp",    # orient hand toward index MCP
            "right_hand": "right_index_mcp",
        }

        targets = {}
        for name in joint_names_ordered:
            lm_name = IK_TO_LANDMARK.get(name, name)
            if lm_name is not None and lm_name in mp_lm:
                targets[name] = mp_lm[lm_name]
            elif name == "pelvis":
                targets[name] = hip_center.copy()

        # Fallback for hands: if index_mcp landmark is collapsed (same as wrist),
        # fall back to wrist target so hand extends from forearm
        for hand_name, fallback_lm in [("left_hand", "left_wrist"), ("right_hand", "right_wrist")]:
            if hand_name in targets and fallback_lm in mp_lm:
                if np.allclose(targets[hand_name], mp_lm[fallback_lm], atol=1e-6):
                    targets[hand_name] = mp_lm[fallback_lm]

        # Run IK
        ik.solve(targets, iterations=15, tolerance=1e-3)

        # Extract pose parameters (axis-angle format)
        # SMPL-X pose: 72 values = 3 (global_orient) + 63 (body_pose) + 6 (hand_pose minimal)
        pose = np.zeros(72)

        # Global orientation: align root to world
        pelvis_pos = ik.joint_positions[name_to_idx["pelvis"]]
        pose[0:3] = [0.0, 0.0, 0.0]  # T-pose global orient

        # Body pose: convert joint rotations to axis-angle
        # Body joints in SMPL-X order (excluding root):
        # 1: left_hip, 2: right_hip, 3: spine_1, 4: left_knee, 5: right_knee,
        # 6: spine_2, 7: left_ankle, 8: right_ankle, 9: spine_3,
        # 10: left_foot_side, 11: right_foot_side, 12: neck,
        # 13: left_clavicle, 14: right_clavicle, 15: head,
        # 16: left_upper_arm, 17: right_upper_arm,
        # 18: left_elbow, 19: right_elbow,
        # 20: left_forearm, 21: right_forearm,
        # 22: left_hand, 23: right_hand
        body_joint_order = [
            "left_hip", "right_hip", "spine_1",
            "left_knee", "right_knee", "spine_2",
            "left_ankle", "right_ankle", "spine_3",
            "left_foot", "right_foot",               # joints 10, 11 — must be present
            "neck", "left_clavicle", "right_clavicle", "head",
            "left_upper_arm", "right_upper_arm",
            "left_forearm", "right_forearm",
            "left_hand", "right_hand",
        ]

        # Map each body joint name to its SMPL-X body_pose index (joints 1-21).
        # body_pose starts at pose[3], each joint occupies 3 values:
        #   pose[3 + (smpl_idx - 1) * 3 : 3 + smpl_idx * 3]
        JOINT_TO_SMPL_BODY_IDX = {
            "left_hip": 1, "right_hip": 2, "spine_1": 3,
            "left_knee": 4, "right_knee": 5, "spine_2": 6,
            "left_ankle": 7, "right_ankle": 8, "spine_3": 9,
            "left_foot": 10, "right_foot": 11,
            "neck": 12, "left_clavicle": 13, "right_clavicle": 14, "head": 15,
            "left_upper_arm": 16, "right_upper_arm": 17,
            "left_forearm": 18, "right_forearm": 19,
            "left_hand": 20, "right_hand": 21,
        }

        for jname in body_joint_order:
            smpl_idx = JOINT_TO_SMPL_BODY_IDX.get(jname)
            if smpl_idx is None or smpl_idx >= 22:
                continue
            if jname in name_to_idx:
                quat = ik.joint_rotations[name_to_idx[jname]]
                # Ensure shortest-path rotation: if w < 0, negate the quaternion
                # to get an angle in [0, π] instead of [π, 2π]
                if quat[3] < 0:
                    quat = -quat
                # Convert quaternion [x,y,z,w] to axis-angle
                angle = 2 * np.arccos(np.clip(quat[3], -1, 1))
                if abs(angle) < 1e-6:
                    aa = np.array([0.0, 0.0, 0.0])
                else:
                    s = np.sin(angle / 2)
                    if abs(s) < 1e-10:
                        aa = np.array([0.0, 0.0, 0.0])
                    else:
                        aa = np.array([quat[0] / s, quat[1] / s, quat[2] / s]) * angle
                pose_idx = smpl_idx * 3
                if pose_idx + 3 <= 72:
                    pose[pose_idx:pose_idx + 3] = aa

        # Hand pose: compute as a LOCAL-space rotation relative to the hand joint.
        # The direction from wrist→index MCP is in world coordinates; we transform it
        # into the hand's local frame using the hand joint's body rotation, then encode
        # as axis-angle. This avoids the "curling effect" of applying world-space axes
        # as local-space rotations.
        left_index_mcp = mp_lm.get("left_index_mcp", None)
        right_index_mcp = mp_lm.get("right_index_mcp", None)

        # Hand pose at indices 66-71 (6 values for minimal hand pose)
        left_hand_idx = name_to_idx.get("left_hand")
        right_hand_idx = name_to_idx.get("right_hand")
        if left_index_mcp is not None and l_wrist is not None and left_hand_idx is not None:
            world_dir = left_index_mcp - l_wrist
            norm = np.linalg.norm(world_dir)
            if norm > 1e-6:
                world_dir = world_dir / norm
                hand_body_quat = ik.joint_rotations[left_hand_idx]  # [x, y, z, w]
                # Transform world-space direction to hand's local frame:
                # local_dir = conjugate(quat) * world_dir * quat
                local_dir = rotate_vector_by_inverse_quat(world_dir, hand_body_quat)
                pose[66:69] = local_dir * 0.1
        if right_index_mcp is not None and r_wrist is not None and right_hand_idx is not None:
            world_dir = right_index_mcp - r_wrist
            norm = np.linalg.norm(world_dir)
            if norm > 1e-6:
                world_dir = world_dir / norm
                hand_body_quat = ik.joint_rotations[right_hand_idx]
                local_dir = rotate_vector_by_inverse_quat(world_dir, hand_body_quat)
                pose[69:72] = local_dir * 0.1

        transl = pelvis_pos.tolist()
        global_orient = pose[0:3].tolist()

        # Compute SMPL-X joint positions (approximate)
        smplx_joints = np.zeros((128, 3))
        for i, name in enumerate(joint_names_ordered):
            if i < 128:
                smplx_joints[i] = ik.joint_positions[name_to_idx[name]]

        # Compute confidence based on how well IK targets were met
        total_error = 0.0
        count = 0
        for name, target in targets.items():
            if name in name_to_idx:
                eff = ik.joint_positions[name_to_idx[name]]
                total_error += np.linalg.norm(target - eff)
                count += 1
        mean_error = total_error / max(count, 1)
        confidence = max(0.0, 1.0 - mean_error * 2)

        return FitResult(
            pose=pose.tolist(),
            betas=[0.0] * 10,
            transl=transl,
            global_orient=global_orient,
            smplx_joints=smplx_joints.flatten().tolist(),
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# SMPL-X-to-Three.js conversion utilities
# ---------------------------------------------------------------------------

# SMPL-X joint indices that correspond to the Xbot avatar bones
# This mirrors the BONE_MAP in SignAvatar.tsx
SMPLX_TO_AVATAR_BONE_MAP: dict[str, str] = {
    "pelvis":             "root",
    "spine_1":            "torso",
    "spine_3":            "torso",
    "neck":                "neck",
    "head_top":            "head",
    "head":                "head",
    "left_clavicle":       "leftShoulder",
    "right_clavicle":      "rightShoulder",
    "left_upper_arm":      "leftUpperArm",
    "right_upper_arm":     "rightUpperArm",
    "left_forearm":        "leftForearm",
    "right_forearm":       "rightForearm",
    "left_hand":           "leftHand",
    "right_hand":          "rightHand",
}

# SMPL-X joint index to name mapping (official order)
SMPLX_JOINT_INDEX_TO_NAME = {
    0:  "pelvis", 1:  "left_hip", 2:  "right_hip", 3:  "spine_1",
    4:  "left_knee", 5:  "right_knee", 6:  "spine_2",
    7:  "left_ankle", 8:  "right_ankle", 9:  "spine_3",
    10: "left_foot", 11: "right_foot", 12: "neck",
    13: "left_clavicle", 14: "right_clavicle", 15: "head",
    16: "left_upper_arm", 17: "right_upper_arm",
    18: "left_forearm", 19: "right_forearm",
    20: "left_hand", 21: "right_hand",
}

# Hand joint indices in SMPL-X (24-53)
SMPLX_HAND_JOINT_NAMES = [
    "thumb_1", "thumb_2", "thumb_3",
    "index_1", "index_2", "index_3",
    "middle_1", "middle_2", "middle_3",
    "ring_1", "ring_2", "ring_3",
    "pinky_1", "pinky_2", "pinky_3",
]


def smplx_pose_to_avatar_quats(
    pose: list[float],
    parent_world_quats: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Convert SMPL-X pose parameters to avatar bone quaternions.

    Args:
        pose: 72 floats (global_orient + body_pose + hand_pose)
        parent_world_quats: Parent bone world quaternions for chain computation

    Returns:
        Dictionary mapping avatar bone keys to quaternion [x, y, z, w]
    """
    pose_arr = np.asarray(pose, dtype=np.float64)

    # Parse SMPL-X pose into joint axis-angle rotations
    # pose[0:3]   = global_orient (root rotation)
    # pose[3:66]  = body_pose (21 joints × 3, including root as #0)
    # pose[66:72] = left hand pose (3) + right hand pose (3)
    global_orient = pose_arr[0:3]
    body_pose = pose_arr[3:66]  # 63 values = 21 joints
    hand_pose = pose_arr[66:72]  # 6 values = 2 per hand

    # Convert each joint's axis-angle to quaternion [x, y, z, w]
    joint_quats: dict[int, np.ndarray] = {}
    joint_quats[0] = rotvec_to_quat(global_orient)

    for i in range(1, 22):
        idx = i * 3
        if idx + 3 <= len(body_pose):
            joint_quats[i] = rotvec_to_quat(body_pose[idx - 3:idx])

    # Map SMPL-X joint quaternions to avatar bone quaternions
    result: dict[str, list[float]] = {}

    for smpl_idx, joint_name in SMPLX_JOINT_INDEX_TO_NAME.items():
        if smpl_idx in joint_quats:
            av_key = SMPLX_TO_AVATAR_BONE_MAP.get(joint_name)
            if av_key and av_key not in result:
                q = joint_quats[smpl_idx]
                result[av_key] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

    # Hand orientation
    if "leftHand" in SMPLX_TO_AVATAR_BONE_MAP.values():
        lh_q = rotvec_to_quat(hand_pose[0:3])
        result["leftHand"] = [float(lh_q[0]), float(lh_q[1]), float(lh_q[2]), float(lh_q[3])]
    if "rightHand" in SMPLX_TO_AVATAR_BONE_MAP.values():
        rh_q = rotvec_to_quat(hand_pose[3:6])
        result["rightHand"] = [float(rh_q[0]), float(rh_q[1]), float(rh_q[2]), float(rh_q[3])]

    return result
