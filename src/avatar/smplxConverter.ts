import type { AvatarFrame } from "./retarget";

/**
 * SMPL-X pose parameter layout:
 *   [0..2]   global_orient  (axis-angle, 3 floats)
 *   [3..65]  body_pose      (21 joints × 3 = 63 floats, joint 0 = root reiterated)
 *   [66..71] hand_pose      (left: 3, right: 3)
 * Total: 72 floats per frame
 *
 * These are axis-angle vectors where direction = rotation axis and magnitude = angle (radians).
 */

export interface SmplxPose {
  pose: number[];        // 72 floats
  betas: number[];       // 10 floats (shape parameters)
  transl: number[];      // 3 floats (root translation)
  global_orient: number[]; // 3 floats
  smplx_joints: number[]; // 128 × 3 = 384 floats
  confidence: number;
}

export interface SmplxFrame {
  label: string;
  frames: SmplxPose[];
  frameRate: number;
}

export const SMPLX_POSE_LENGTH = 72;
export const SMPLX_BETA_LENGTH = 10;
export const SMPLX_JOINT_COUNT = 128;

// Bone frame correction quaternions: transforms SMPL-X body_pose rotations
// (computed by the IK solver relative to T-pose initial directions) to
// Xbot GLTF local bone rotations (which start from Xbot bind-pose default
// bone directions).
//
// The correct frame transformation is: result = R * q * R^(-1) (conjugation),
// where R maps the IK T-pose bone direction to the Xbot bind-pose direction.
//
// IK T-pose directions vs Xbot defaults (from retarget.ts DEFAULT_BONE_DIRS):
//   Bone         | IK initial    | Xbot default   | Correction quat [x,y,z,w]
//   leftUpperArm | [-1, 0, 0]    |  [1, 0, 0]    | [0, 0, 1, 0]    (180° Z)
//   rightUpperArm| [1, 0, 0]     | [-1, 0, 0]    | [0, 0, -1, 0]   (180° Z)
//   leftForearm  | [0, -1, 0]    |  [1, 0, 0]    | [0, 0, √2/2, √2/2]  (+90° Z)
//   rightForearm | [0, -1, 0]    | [-1, 0, 0]    | [0, 0, -√2/2, √2/2] (-90° Z)
//   leftHand     | [0, -1, 0]    |  [1, 0, 0]    | [0, 0, √2/2, √2/2]  (+90° Z)
//   rightHand    | [0, -1, 0]    | [-1, 0, 0]    | [0, 0, -√2/2, √2/2] (-90° Z)
//   neck         | [0, 1, 0]     |  [0, 0, 1]    | [√2/2, 0, 0, √2/2]  (+90° X)
//   head         | [0, 1, 0]     |  [0, 0, 1]    | [√2/2, 0, 0, √2/2]  (+90° X)
//   torso        | [0, 1, 0]     |  [0, 1, 0]    | identity
//   leftShoulder | dynamic       | varies        | identity
//   rightShoulder| dynamic       | varies        | identity
//
// Frame transformation: result = correction * q * conjugate(correction)
const BONE_FRAME_CORRECTION: Record<string, [number, number, number, number]> = {
  leftShoulder: [0, 0, 0, 1],
  rightShoulder: [0, 0, 0, 1],
  leftUpperArm: [0, 0, 1, 0],
  rightUpperArm: [0, 0, -1, 0],
  leftForearm: [0, 0, 0.707107, 0.707107],
  rightForearm: [0, 0, -0.707107, 0.707107],
  leftHand: [0, 0, 0.707107, 0.707107],
  rightHand: [0, 0, -0.707107, 0.707107],
  torso: [0, 0, 0, 1],
  neck: [0.707107, 0, 0, 0.707107],
  head: [0.707107, 0, 0, 0.707107],
};



/**
 * Convert an axis-angle vector to a quaternion [x, y, z, w].
 * Axis-angle: direction = axis, magnitude = angle in radians.
 */
export function axisAngleToQuat(aa: [number, number, number]): [number, number, number, number] {
  const angle = Math.hypot(aa[0], aa[1], aa[2]);
  if (angle < 1e-10) return [0, 0, 0, 1];
  const s = Math.sin(angle / 2) / angle;
  return [aa[0] * s, aa[1] * s, aa[2] * s, Math.cos(angle / 2)];
}

/**
 * Quaternion multiplication: returns a * b (applied right-to-left).
 * Both inputs are [x, y, z, w].
 */
export function multiplyQuats(a: number[], b: number[]): number[] {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/**
 * Conjugate of a quaternion [x, y, z, w] -> [-x, -y, -z, w].
 * For unit quaternions, this is the multiplicative inverse.
 */
function conjugateQuat(q: number[]): number[] {
  return [-q[0], -q[1], -q[2], q[3]];
}

/**
 * Transform a rotation from one reference frame to another.
 * result = R * q * R^(-1), where R is the frame change quaternion.
 * This preserves the rotation angle while changing the axis orientation.
 */
function transformQuatByFrame(q: number[], frame: number[]): number[] {
  const frameConj = conjugateQuat(frame);
  return multiplyQuats(frame, multiplyQuats(q, frameConj));
}



/**
 * Convert SMPL-X pose parameters to an AvatarFrame compatible with SignAvatar.tsx.
 *
 * SMPL-X body_pose indices (after the 3 global_orient floats):
 *   index 0  = joint 0 (root/pelvis) — usually identity in body_pose
 *   index 1  = joint 1 (left_hip)
 *   index 2  = joint 2 (right_hip)
 *   ...
 *   index 16 = joint 16 (left_upper_arm)
 *   index 17 = joint 17 (right_upper_arm)
 *   index 18 = joint 18 (left_elbow) — not in standard SMPL-X body_pose
 *   index 19 = joint 19 (right_elbow) — not in standard SMPL-X body_pose
 *   index 20 = joint 20 (left_forearm)
 *   index 21 = joint 21 (right_forearm)
 *   index 22 = joint 22 (left_hand)
 *   index 23 = joint 23 (right_hand)
 *
 * The body_pose is 63 floats = 21 joints × 3 (joints 1..21, excluding root at index 0).
 */
export function smplxPoseToAvatarFrame(
  smplxPose: SmplxPose,
  parentWorldQuats?: Record<string, [number, number, number, number]>,
): AvatarFrame {
  void parentWorldQuats; // Reserved for parent-space chain computation in full implementation
  const pose = smplxPose.pose;

  // Global orientation (root rotation) is encoded in SMPL-X pose[0:3] as axis-angle.
  // It applies to the root/hips and is used as the starting point for the body chain.
  // Initialize the frame with identity/position defaults
  const frame: AvatarFrame = {
    root: [0, 0, 0],
    hips: [0, 0, 0],
    torso: [0, 0, 0, 1],
    head: [0, 0, 0, 1],
    neck: [0, 0, 0, 1],
    leftShoulder: [0, 0, 0, 1],
    leftUpperArm: [0, 0, 0, 1],
    leftForearm: [0, 0, 0, 1],
    leftHand: [0, 0, 0, 1],
    rightShoulder: [0, 0, 0, 1],
    rightUpperArm: [0, 0, 0, 1],
    rightForearm: [0, 0, 0, 1],
    rightHand: [0, 0, 0, 1],
  };

  // The body_pose starts at index 3 in the 72-dim pose vector.
  // body_pose[0:3] corresponds to SMPL-X joint 1 (left_hip), body_pose[1:3×2] to joint 2, etc.
  // SMPL-X joint index = body_pose joint index + 1 (because root is at index 0, separate)
  // body_pose index = (smplx_joint_idx - 1) * 3 + 3

   function getBodyJointQuat(smplxJointIdx: number, avatarKey: string): [number, number, number, number] {
     // body_pose starts at pose[3], each joint is 3 floats
     // smplxJointIdx 1 → body_pose offset 0
     // smplxJointIdx 16 → body_pose offset (16-1)*3 = 45, so pose[3+45] = pose[48]
     const offset = 3 + (smplxJointIdx - 1) * 3;
     const aa: [number, number, number] = [pose[offset], pose[offset + 1], pose[offset + 2]];
     const q = axisAngleToQuat(aa);
     // Apply bone frame correction: result = R * q * R^(-1)
     const correction = BONE_FRAME_CORRECTION[avatarKey] || [0, 0, 0, 1];
     return transformQuatByFrame(q, correction) as [number, number, number, number];
   }

  // Map SMPL-X joint rotations to avatar bones
  // Note: SMPL-X rotations are in the SMPL-X local frame; we need to convert
  // to the Xbot GLTF local frame. The parent world quaternions help bridge this.

   // Torso (spine): SMPL-X joint 9 (spine_3) - axis-angle to quaternion
   frame.torso = [...getBodyJointQuat(9, "torso")] as [number, number, number, number];

   // Head: SMPL-X joint 15
  frame.head = [...getBodyJointQuat(15, "head")] as [number, number, number, number];

  // Neck: SMPL-X joint 12
  frame.neck = [...getBodyJointQuat(12, "neck")] as [number, number, number, number];

  // Left shoulder (clavicle): SMPL-X joint 13
  frame.leftShoulder = [...getBodyJointQuat(13, "leftShoulder")] as [number, number, number, number];

  // Right shoulder (clavicle): SMPL-X joint 14
  frame.rightShoulder = [...getBodyJointQuat(14, "rightShoulder")] as [number, number, number, number];

  // Left upper arm: SMPL-X joint 16
  frame.leftUpperArm = [...getBodyJointQuat(16, "leftUpperArm")] as [number, number, number, number];

  // Right upper arm: SMPL-X joint 17
  frame.rightUpperArm = [...getBodyJointQuat(17, "rightUpperArm")] as [number, number, number, number];

    // Left forearm: SMPL-X joint 18 (left_forearm)
    frame.leftForearm = [...getBodyJointQuat(18, "leftForearm")] as [number, number, number, number];

    // Right forearm: SMPL-X joint 19 (right_forearm)
    frame.rightForearm = [...getBodyJointQuat(19, "rightForearm")] as [number, number, number, number];

    // Left hand: SMPL-X joint 20 (left_hand) + hand_pose (pose[66:69])
    // Combine body_pose hand rotation with hand_pose in SMPL-X frame, THEN apply frame correction.
    {
      const offset = 3 + (20 - 1) * 3;
      const lhAA: [number, number, number] = [pose[offset], pose[offset + 1], pose[offset + 2]];
      const leftHandAA: [number, number, number] = [pose[66], pose[67], pose[68]];
      const lhBodyQuat = axisAngleToQuat(lhAA);
      const leftHandPoseQuat = axisAngleToQuat(leftHandAA);
      // Combine in SMPL-X local frame: body_quat * hand_pose (apply hand_pose first, then body)
      const lhCombined = multiplyQuats(lhBodyQuat, leftHandPoseQuat);
      const lhCorrection = BONE_FRAME_CORRECTION["leftHand"] || [0, 0, 0, 1];
      // Apply frame correction: R * combined * R^(-1)
      frame.leftHand = transformQuatByFrame(lhCombined, lhCorrection) as [number, number, number, number];
    }

    // Right hand: SMPL-X joint 21 (right_hand) + hand_pose (pose[69:72])
    {
      const offset = 3 + (21 - 1) * 3;
      const rhAA: [number, number, number] = [pose[offset], pose[offset + 1], pose[offset + 2]];
      const rightHandAA: [number, number, number] = [pose[69], pose[70], pose[71]];
      const rhBodyQuat = axisAngleToQuat(rhAA);
      const rightHandPoseQuat = axisAngleToQuat(rightHandAA);
      const rhCombined = multiplyQuats(rhBodyQuat, rightHandPoseQuat);
      const rhCorrection = BONE_FRAME_CORRECTION["rightHand"] || [0, 0, 0, 1];
      frame.rightHand = transformQuatByFrame(rhCombined, rhCorrection) as [number, number, number, number];
    }

  // Root position
  if (smplxPose.transl && smplxPose.transl.length >= 3) {
    frame.root = [smplxPose.transl[0], smplxPose.transl[1], smplxPose.transl[2]];
    frame.hips = frame.root;
  }

  return frame;
}

/**
 * Convert a sequence of SMPL-X poses to AvatarFrames.
 */
export function smplxToAvatarFrames(
  poses: SmplxPose[],
  parentWorldQuats?: Record<string, [number, number, number, number]>,
): AvatarFrame[] {
  return poses.map((pose) => smplxPoseToAvatarFrame(pose, parentWorldQuats));
}

/**
 * Fetch SMPL-X poses from the signavatars-service.
 * The service returns pre-computed SMPL-X parameters for cached labels,
 * or computes them on-the-fly from MediaPipe landmarks.
 */
export async function fetchSmplxFromService(
  label: string,
  options?: {
    landmarks?: number[];
    parentWorldQuats?: Record<string, [number, number, number, number]>;
    baseUrl?: string;
  },
): Promise<AvatarFrame[] | null> {
  const baseUrl = options?.baseUrl || "http://localhost:8000";
  const endpoint = `/fit-sequence`;

  try {
    if (options?.landmarks) {
      // Live fitting: send landmarks to the service
      const resp = await fetch(`${baseUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label,
          frames: [options.landmarks],
          parent_world_quats: options.parentWorldQuats,
        }),
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      const poses: SmplxPose[] = data.frames || [];
      return smplxToAvatarFrames(poses, options.parentWorldQuats);
    } else {
      // Fetch pre-computed SMPL-X data (if available from the service)
      const resp = await fetch(`${baseUrl}/avatar-data-smplx/${encodeURIComponent(label)}.json`);
      if (!resp.ok) return null;
      const data = await resp.json() as SmplxFrame;
      return smplxToAvatarFrames(data.frames, options?.parentWorldQuats);
    }
  } catch (e) {
    console.error("[smplx] fetch failed:", e);
    return null;
  }
}
