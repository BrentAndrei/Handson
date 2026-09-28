"""
End-to-end validation: trace SMPL-X pose data through the TypeScript converter
logic (reproduced in Python) to verify bone rotations are correct.
"""
import json, math

def axis_angle_to_quat(aa):
    angle = math.sqrt(aa[0]**2 + aa[1]**2 + aa[2]**2)
    if angle < 1e-10:
        return [0.0, 0.0, 0.0, 1.0]
    s = math.sin(angle / 2) / angle
    return [aa[0] * s, aa[1] * s, aa[2] * s, math.cos(angle / 2)]

def multiply_quats(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]

def conjugate_quat(q):
    return [-q[0], -q[1], -q[2], q[3]]

def transform_by_frame(q, frame):
    """R * q * R^(-1) — correct frame transformation"""
    frame_conj = conjugate_quat(frame)
    return multiply_quats(frame, multiply_quats(q, frame_conj))

BONE_FRAME_CORRECTION = {
    "leftShoulder": [0, 0, 0, 1],
    "rightShoulder": [0, 0, 0, 1],
    "leftUpperArm": [0, 0, 1, 0],
    "rightUpperArm": [0, 0, -1, 0],
    "leftForearm": [0, 0, 0.707107, 0.707107],
    "rightForearm": [0, 0, -0.707107, 0.707107],
    "leftHand": [0, 0, 0.707107, 0.707107],
    "rightHand": [0, 0, -0.707107, 0.707107],
    "torso": [0, 0, 0, 1],
    "neck": [0.707107, 0, 0, 0.707107],
    "head": [0.707107, 0, 0, 0.707107],
}

# Xbot bind-pose bone directions (from retarget.ts DEFAULT_BONE_DIRS)
XBOT_DIRS = {
    "leftUpperArm": [1, 0, 0],
    "rightUpperArm": [-1, 0, 0],
    "leftForearm": [1, 0, 0],
    "rightForearm": [-1, 0, 0],
    "leftHand": [1, 0, 0],
    "rightHand": [-1, 0, 0],
    "neck": [0, 0, 1],
    "head": [0, 0, 1],
    "torso": [0, 1, 0],
}

# IK T-pose directions
IK_DIRS = {
    "leftUpperArm": [-1, 0, 0],
    "rightUpperArm": [1, 0, 0],
    "leftForearm": [0, -1, 0],
    "rightForearm": [0, -1, 0],
    "leftHand": [0, -1, 0],
    "rightHand": [0, -1, 0],
    "neck": [0, 1, 0],
    "head": [0, 1, 0],
    "torso": [0, 1, 0],
}

def quat_to_axis_angle(q):
    w = q[3]
    if w > 1: w = 1
    if w < -1: w = -1
    angle = 2 * math.acos(abs(w))  # shortest path
    if angle < 1e-6:
        return [0, 0, 0]
    s = math.sin(angle / 2)
    return [q[0] / s * angle, q[1] / s * angle, q[2] / s * angle]

def quat_rotate_vector(q, v):
    """Rotate vector v by quaternion q (v as pure quaternion)"""
    q_conj = conjugate_quat(q)
    vq = [v[0], v[1], v[2], 0.0]
    result = multiply_quats(multiply_quats(q, vq), q_conj)
    return [result[0], result[1], result[2]]

# Test: verify correction quaternions map IK dirs to Xbot dirs
print("=== Verifying correction quaternions map IK T-pose to Xbot bind-pose ===")
for bone in IK_DIRS:
    ik_dir = IK_DIRS[bone]
    xbot_dir = XBOT_DIRS[bone]
    R = BONE_FRAME_CORRECTION[bone]
    rotated = quat_rotate_vector(R, ik_dir)
    match = all(abs(rotated[i] - xbot_dir[i]) < 1e-4 for i in range(3))
    status = "OK" if match else "FAIL"
    print(f"  {bone}: IK={ik_dir} → Xbot={xbot_dir}, R*{ik_dir}={list(round(x,4) for x in rotated)} [{status}]")

# Test: verify identity rotation stays identity after correction
print("\n=== Verifying identity rotation stays identity ===")
for bone in ["leftUpperArm", "leftForearm", "leftHand", "neck", "head", "torso"]:
    q = [0, 0, 0, 1]  # identity
    R = BONE_FRAME_CORRECTION[bone]
    result = transform_by_frame(q, R)
    angle = math.degrees(2 * math.acos(max(-1, min(1, abs(result[3])))))
    status = "OK" if angle < 1 else "FAIL"
    print(f"  {bone}: angle={angle:.2f} deg [{status}]")

# Test: trace ADOPT sign through converter
print("\n=== ADOPT frame 0 conversion ===")
with open("public/avatar-data-smplx/ADOPT.smplx.json", "r") as f:
    data = json.load(f)
pose = data["frames"][0]["pose"]

joints = {
    "torso": (9, "torso"),
    "neck": (12, "neck"),
    "head": (15, "head"),
    "leftUpperArm": (16, "leftUpperArm"),
    "rightUpperArm": (17, "rightUpperArm"),
    "leftForearm": (18, "leftForearm"),
    "rightForearm": (19, "rightForearm"),
}

print("  Bone rotations after frame correction (R * q * R^(-1)):")
for name, (smpl_idx, bone_key) in joints.items():
    offset = 3 + (smpl_idx - 1) * 3
    aa = pose[offset:offset+3]
    q = axis_angle_to_quat(aa)
    R = BONE_FRAME_CORRECTION[bone_key]
    result = transform_by_frame(q, R)
    angle = math.degrees(2 * math.acos(max(-1, min(1, abs(result[3])))))
    print(f"    {name}: IK angle={math.degrees(math.sqrt(sum(x*x for x in aa))):.1f}° → Xbot angle={angle:.1f}°")

# Test hand combination
print("\n=== Hand combination (body_pose * hand_pose, then R * q * R^(-1)) ===")
for side, smpl_idx, pose_idx in [("left", 20, 66), ("right", 21, 69)]:
    offset = 3 + (smpl_idx - 1) * 3
    body_aa = pose[offset:offset+3]
    hand_aa = pose[pose_idx:pose_idx+3]
    body_q = axis_angle_to_quat(body_aa)
    hand_q = axis_angleToQuat = axis_angle_to_quat(hand_aa)
    combined = multiply_quats(body_q, hand_q)
    R = BONE_FRAME_CORRECTION[f"{side}Hand"]
    result = transform_by_frame(combined, R)
    angle = math.degrees(2 * math.acos(max(-1, min(1, abs(result[3])))))
    body_angle = math.degrees(math.sqrt(sum(x*x for x in body_aa)))
    print(f"  {side}Hand: body_angle={body_angle:.1f}° + hand_pose={math.degrees(math.sqrt(sum(x*x for x in hand_aa))):.1f}° → result={angle:.1f}°")

# Final angle check across ALL frames
print("\n=== Final: checking all angles <= 180° across all frames ===")
import os
smplx_dir = "public/avatar-data-smplx"
files = sorted(f for f in os.listdir(smplx_dir) if f.endswith(".smplx.json"))
total = 0
max_angle = 0
over_180 = 0
for fname in files:
    with open(os.path.join(smplx_dir, fname)) as f:
        data = json.load(f)
    for frame in data["frames"]:
        pose = frame["pose"]
        total += 1
        for i in range(1, 22):
            offset = 3 + (i - 1) * 3
            aa = pose[offset:offset+3]
            angle = math.degrees(math.sqrt(sum(x*x for x in aa)))
            if angle > max_angle:
                max_angle = angle
            if angle > 180:
                over_180 += 1
print(f"  Files: {len(files)}, Frames: {total}, Max angle: {max_angle:.1f}°, Over 180°: {over_180}")
