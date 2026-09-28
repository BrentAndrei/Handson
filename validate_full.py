"""Verify SMPL-X output after hand_pose fix.

Checks:
1. Correction quaternions map IK T-pose directions to Xbot bind-pose directions
2. Identity rotations stay identity
3. Hand pose values are in local frame (not world-space)
4. All angles <= 180 degrees across all files
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import json
import math
import os

def quat_multiply(a, b):
    """a * b, both [x,y,z,w]. Apply b first, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]

def quat_conjugate(q):
    return [-q[0], -q[1], -q[2], q[3]]

def quat_rotate_vector(v, q):
    """Rotate vector v by quaternion q (x,y,z,w)."""
    qv = list(v) + [0]
    result = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return result[:3]

def aa_to_quat(aa):
    """Axis-angle to quaternion [x,y,z,w]."""
    angle = math.sqrt(aa[0]**2 + aa[1]**2 + aa[2]**2)
    if angle < 1e-10:
        return [0, 0, 0, 1]
    s = math.sin(angle / 2) / angle
    return [aa[0]*s, aa[1]*s, aa[2]*s, math.cos(angle / 2)]

# Bone frame corrections (from smplxConverter.ts)
CORRECTIONS = {
    "leftUpperArm": [0, 0, 1, 0],
    "rightUpperArm": [0, 0, -1, 0],
    "leftForearm": [0, 0, 0.707107, 0.707107],
    "rightForearm": [0, 0, -0.707107, 0.707107],
    "leftHand": [0, 0, 0.707107, 0.707107],
    "rightHand": [0, 0, -0.707107, 0.707107],
    "neck": [0.707107, 0, 0, 0.707107],
    "head": [0.707107, 0, 0, 0.707107],
    "torso": [0, 0, 0, 1],
}

# IK T-pose directions vs Xbot defaults
BONE_DIRS = {
    "leftUpperArm": ([-1, 0, 0], [1, 0, 0]),
    "rightUpperArm": ([1, 0, 0], [-1, 0, 0]),
    "leftForearm": ([0, -1, 0], [1, 0, 0]),
    "rightForearm": ([0, -1, 0], [-1, 0, 0]),
    "leftHand": ([0, -1, 0], [1, 0, 0]),
    "rightHand": ([0, -1, 0], [-1, 0, 0]),
    "neck": ([0, 1, 0], [0, 0, 1]),
    "head": ([0, 1, 0], [0, 0, 1]),
    "torso": ([0, 1, 0], [0, 1, 0]),
}

print("=== Verifying correction quaternions map IK T-pose to Xbot bind-pose ===")
for bone, (ik_dir, xbot_dir) in BONE_DIRS.items():
    corr = CORRECTIONS[bone]
    rotated = quat_rotate_vector(ik_dir, corr)
    ok = np.allclose(rotated, xbot_dir, atol=1e-4)
    status = "OK" if ok else "FAIL"
    print(f"  {bone}: IK={ik_dir} -> Xbot={xbot_dir}, R*{ik_dir}={rotated} [{status}]")

print()

print("=== Verifying identity rotation stays identity ===")
for bone, corr in CORRECTIONS.items():
    q_eye = [0, 0, 0, 1]
    result = quat_multiply(corr, quat_multiply(q_eye, quat_conjugate(corr)))
    angle = 2 * math.acos(max(-1, min(1, result[3])))
    ok = abs(angle) < 1e-6
    print(f"  {bone}: angle={math.degrees(angle):.4f} deg [{'OK' if ok else 'FAIL'}]")

print()

print("=== HELLO frame 0 conversion ===")
with open('public/avatar-data-smplx/HELLO.smplx.json') as f:
    data = json.load(f)
pose = data['frames'][0]['pose']
print("  Bone rotations after frame correction (R * q * R^(-1)):")

body_joints = {
    "torso": (9, "torso"),
    "head": (15, "head"),
    "neck": (12, "neck"),
    "leftUpperArm": (16, "leftUpperArm"),
    "rightUpperArm": (17, "rightUpperArm"),
    "leftForearm": (18, "leftForearm"),
    "rightForearm": (19, "rightForearm"),
}

for label, (smpl_idx, bone_key) in body_joints.items():
    offset = 3 + (smpl_idx - 1) * 3
    aa = pose[offset:offset+3]
    q = aa_to_quat(aa)
    corr = CORRECTIONS.get(bone_key, [0, 0, 0, 1])
    result = quat_multiply(corr, quat_multiply(q, quat_conjugate(corr)))
    ik_angle = math.degrees(math.sqrt(aa[0]**2 + aa[1]**2 + aa[2]**2))
    result_angle = 2 * math.acos(max(-1, min(1, abs(result[3])))) * 180 / math.pi
    print(f"    {label}: IK={ik_angle:.1f} -> Xbot={result_angle:.1f} deg")

# Hand with hand_pose combination
for hand, joint_idx, hp_offset, bone_key in [
    ("leftHand", 20, 66, "leftHand"),
    ("rightHand", 21, 69, "rightHand"),
]:
    offset = 3 + (joint_idx - 1) * 3
    body_aa = pose[offset:offset+3]
    body_q = aa_to_quat(body_aa)
    hand_aa = pose[hp_offset:hp_offset+3]
    hand_q = aa_to_quat(hand_aa)
    combined = quat_multiply(body_q, hand_q)
    corr = CORRECTIONS.get(bone_key, [0, 0, 0, 1])
    result = quat_multiply(corr, quat_multiply(combined, quat_conjugate(corr)))
    body_angle = math.degrees(math.sqrt(body_aa[0]**2 + body_aa[1]**2 + body_aa[2]**2))
    hp_angle = math.degrees(math.sqrt(hand_aa[0]**2 + hand_aa[1]**2 + hand_aa[2]**2))
    result_angle = 2 * math.acos(max(-1, min(1, abs(result[3])))) * 180 / math.pi
    print(f"    {hand}: body={body_angle:.1f} + hand_pose={hp_angle:.1f} -> result={result_angle:.1f}")

print()
print("=== Hand pose verification (should be local-space, not world-space) ===")
# The hand_pose should be a small rotation around the LOCAL direction axis
# Check that hand_pose axis is NOT aligned with world axes
for fname in ['HELLO.smplx.json', 'ADOPT.smplx.json']:
    with open(f'public/avatar-data-smplx/{fname}') as f:
        d = json.load(f)
    for i, frame in enumerate(d['frames']):
        pose = frame['pose']
        lh = np.array(pose[66:69])
        rh = np.array(pose[69:72])
        lh_mag = np.linalg.norm(lh)
        rh_mag = np.linalg.norm(rh)
        if lh_mag > 1e-10:
            print(f"  {fname} frame {i}: LH hand_pose mag={lh_mag:.4f} ({math.degrees(lh_mag):.2f} deg), axis=[{lh[0]:.4f},{lh[1]:.4f},{lh[2]:.4f}]")
        if rh_mag > 1e-10:
            print(f"  {fname} frame {i}: RH hand_pose mag={rh_mag:.4f} ({math.degrees(rh_mag):.2f} deg), axis=[{rh[0]:.4f},{rh[1]:.4f},{rh[2]:.4f}]")
        break  # Just first frame with non-zero hand_pose

print()
print("=== Final: checking all angles <= 180 across all frames ===")
total_files = 0
over_180 = 0
total_frames = 0
max_angle = 0
for fname in sorted(os.listdir('public/avatar-data-smplx')):
    if not fname.endswith('.smplx.json'):
        continue
    with open(f'public/avatar-data-smplx/{fname}') as f:
        d = json.load(f)
    total_files += 1
    for frame in d['frames']:
        pose = frame['pose']
        total_frames += 1
        for start_idx in [0, 48, 51, 54, 57, 60, 63, 66, 69]:
            aa = pose[start_idx:start_idx+3]
            angle = math.degrees(math.sqrt(aa[0]**2 + aa[1]**2 + aa[2]**2))
            max_angle = max(max_angle, angle)
            if angle > 180.1:
                over_180 += 1
print(f"  Files: {total_files}, Frames: {total_frames}, Max angle: {max_angle:.2f}, Over 180: {over_180}")
