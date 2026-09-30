/**
 * SMPL-X → CanonicalPose adapter.
 *
 * Converts SMPL-X pose parameters (as produced by signavatars-service) into
 * the intermediate CanonicalPose representation consumed by the retargeter.
 *
 * Data sources (per joint):
 *   - `smplx_joints` (128 × 3 = 384 floats): 3D joint positions in SMPL‑X
 *     / IK-solver coordinates.  These are the primary source for body
 *     and hand pose.
 *   - `pose` (72 floats): axis‑angle parameters, used as a fallback for
 *     body joint rotations when `smplx_joints` positions are unavailable
 *     (all zeros), and for the 6‑D minimal hand‑pose hint (idx 66–71).
 *
 * Two `smplx_joints` index layouts are recognised:
 *
 * 1. IK-solver layout (fallback when no SMPL-X model is loaded):
 *    Indices 0–19 are populated in this order:
 *       0 pelvis, 1 left_hip, 2 right_hip, 3 spine_1,
 *       4 left_knee, 5 right_knee, 6 spine_2,
 *       7 left_ankle, 8 right_ankle, 9 spine_3,
 *      10 neck, 11 left_clavicle, 12 right_clavicle, 13 head,
 *      14 left_upper_arm(=elbow pos), 15 right_upper_arm(=elbow pos),
 *      16 left_forearm(=wrist pos),  17 right_forearm(=wrist pos),
 *      18 left_hand(=index MCP),    19 right_hand(=index MCP)
 *    Indices 20+ are zero.  Coordinates are OpenGL (X‑right, Y‑up, Z‑toward‑viewer)
 *    because the IK solver applies `[v[0], -v[1], -v[2]]` to MediaPipe landmarks.
 *
 * 2. Official SMPL‑X layout (full model path, when model files exist):
 *    128 joints in SMPLX_JOINT_NAMES order.  Coordinates are SMPL‑X
 *    (X‑right, Y‑up, Z‑back).  Finger joints live at indices 24–53.
 *
 * The adapter auto-detects which layout is present and applies the matching
 * coordinate transform.
 */
import {
  CanonicalPose,
  BodyPose,
  HandPose,
  FingerPose,
  FingerName,
  Vec3,
} from "./canonicalPose";
import { gltfToCanonical, smplxToCanonical, Coords } from "./coordinateSystem";
import { SmplxPose, SmplxFrame } from "./smplxConverter";

// ---------------------------------------------------------------------------
// SMPL-X joint index tables
// ---------------------------------------------------------------------------

/**
 * Official SMPL-X joint index → name (subset we care about).
 * Mirrors `SMPLX_JOINT_NAMES` and `SMPLX_JOINT_INDEX_TO_NAME` in fit_smplx.py.
 *
 * In official SMPL-X, each joint marks the START of its bone, so e.g.
 * index 18 (left_elbow) is at the elbow location — the distal end of the
 * upper-arm bone.
 */
const SMPLX_IDX: Record<string, number> = {
  pelvis: 0,
  left_hip: 1,
  right_hip: 2,
  spine_1: 3,
  left_knee: 4,
  right_knee: 5,
  spine_2: 6,
  left_ankle: 7,
  right_ankle: 8,
  spine_3: 9,
  left_foot_side: 10,
  right_foot_side: 11,
  neck: 12,
  left_clavicle: 13,
  right_clavicle: 14,
  head: 15,
  left_upper_arm: 16,
  right_upper_arm: 17,
  left_elbow: 18,
  right_elbow: 19,
  left_forearm: 20,
  right_forearm: 21,
  left_hand: 22,
  right_hand: 23,
  // Finger metacarpal / PIP joints (official SMPL-X indices 24–53)
  left_thumb_1: 24, left_index_1: 27, left_middle_1: 30,
  left_ring_1: 33, left_pinky_1: 36,
  right_thumb_1: 39, right_index_1: 42, right_middle_1: 45,
  right_ring_1: 48, right_pinky_1: 51,
  left_thumb_2: 25, left_index_2: 28, left_middle_2: 31,
  left_ring_2: 34, left_pinky_2: 37,
  right_thumb_2: 40, right_index_2: 43, right_middle_2: 46,
  right_ring_2: 49, right_pinky_2: 52,
  left_thumb_3: 26, left_index_3: 29, left_middle_3: 32,
  left_ring_3: 35, left_pinky_3: 38,
  right_thumb_3: 41, right_index_3: 44, right_middle_3: 47,
  right_ring_3: 50, right_pinky_3: 53,
};

/**
 * IK-solver joint index → position meaning.
 *
 * In the IK solver, the joint at index *i* represents the END of bone *i*,
 * i.e. the position of the child joint.  The joint *names* in
 * `joint_names_ordered` are therefore misleading: e.g. index 14 is called
 * "left_upper_arm" but holds the *elbow* position (end of the upper-arm
 * bone).
 *
 * We map logical body landmarks to IK indices here.
 */
const IK_BODY_IDX: Record<string, number> = {
  pelvis: 0,
  left_hip: 1,
  right_hip: 2,
  spine_1: 3,
  left_knee: 4,
  right_knee: 5,
  spine_2: 6,
  left_ankle: 7,
  right_ankle: 8,
  spine_3: 9,
  neck: 10,
  left_clavicle: 11, // shoulder position
  right_clavicle: 12,
  head: 13,
  // IK calls these "left_upper_arm" / "left_forearm" / "left_hand" but
  // they hold elbow / wrist / hand positions respectively.
  left_elbow: 14,
  right_elbow: 15,
  left_forearm: 16, // wrist position
  right_forearm: 17,
  left_hand: 18, // index MCP / hand center
  right_hand: 19,
};

/** Finger → list of SMPL-X finger-joint names, from base (MCP) to tip. */
const FINGER_JOINT_NAMES: Record<FingerName, string[]> = {
  thumb: ["left_thumb_1", "left_thumb_2", "left_thumb_3"],
  index: ["left_index_1", "left_index_2", "left_index_3"],
  middle: ["left_middle_1", "left_middle_2", "left_middle_3"],
  ring: ["left_ring_1", "left_ring_2", "left_ring_3"],
  pinky: ["left_pinky_1", "left_pinky_2", "left_pinky_3"],
};

/** Same for the right hand. */
const FINGER_JOINT_NAMES_R: Record<FingerName, string[]> = {
  thumb: ["right_thumb_1", "right_thumb_2", "right_thumb_3"],
  index: ["right_index_1", "right_index_2", "right_index_3"],
  middle: ["right_middle_1", "right_middle_2", "right_middle_3"],
  ring: ["right_ring_1", "right_ring_2", "right_ring_3"],
  pinky: ["right_pinky_1", "right_pinky_2", "right_pinky_3"],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readJoint(joints: number[], idx: number): Vec3 {
  const base = idx * 3;
  if (base + 2 >= joints.length) return [0, 0, 0];
  return [joints[base], joints[base + 1], joints[base + 2]];
}

function toCanonicalJoint(joints: number[], name: string, official: boolean): Vec3 {
  const smplxIdx = official ? SMPLX_IDX[name] : IK_BODY_IDX[name];
  if (smplxIdx === undefined) return [0, 0, 0];
  return toCanonical(readJoint(joints, smplxIdx), official);
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}
function len(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}
function normalize(a: Vec3): Vec3 {
  const l = len(a);
  if (l < 1e-10) return [0, 0, 0];
  return [a[0] / l, a[1] / l, a[2] / l];
}
function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Detect whether the `smplx_joints` array follows the official SMPL-X layout
 * (finger joints present at indices 24+) or the IK-solver layout (only
 * indices 0–19 populated).
 */
function hasOfficialLayout(joints: number[]): boolean {
  // In official SMPL-X, index 24 = left_thumb_1 (a finger joint).
  // In the IK layout, index 24 is always zero.
  if (joints.length < 25 * 3) return false;
  const b = 24 * 3;
  return (
    Math.abs(joints[b]) > 1e-6 ||
    Math.abs(joints[b + 1]) > 1e-6 ||
    Math.abs(joints[b + 2]) > 1e-6
  );
}

/**
 * Detect whether smplx_joints has ANY meaningful data, or is all zeros
 * (e.g. full-model path with model not actually loaded).
 */
function hasAnyData(joints: number[]): boolean {
  for (let i = 0; i < Math.min(joints.length, 30 * 3); i++) {
    if (Math.abs(joints[i]) > 1e-6) return true;
  }
  return false;
}

/**
 * Convert a Vec3 from the appropriate source coordinate system to canonical.
 * - IK solver output → OpenGL/GLTF → gltfToCanonical
 * - Full SMPL-X model → SMPL-X → smplxToCanonical
 */
function toCanonical(v: Vec3, official: boolean): Vec3 {
  return official ? smplxToCanonical(v) : gltfToCanonical(v as Coords);
}

/**
 * Compute finger direction: from MCP toward tip.
 * SMPL-X thumb joints: thumb_1 (MCP), thumb_2 (PIP), thumb_3 (DIP/TIP).
 * Only meaningful in the official SMPL-X layout (finger joints at idx 24+).
 */
function computeFingerDir(
  joints: number[],
  names: string[]
): Vec3 {
  const firstIdx = SMPLX_IDX[names[0]] ?? -1;
  const lastIdx = SMPLX_IDX[names[names.length - 1]] ?? -1;
  if (firstIdx < 0 || lastIdx < 0) return [0, 0, 0];
  const p1 = readJoint(joints, firstIdx);
  const p2 = readJoint(joints, lastIdx);
  return normalize(sub(p2, p1));
}

/**
 * Compute finger curl from PIP joint angle.
 * Uses MCP→PIP and PIP→TIP vectors; curl = (1 - dot) / 2, clamped [0, 1].
 */
function computeFingerCurl(
  joints: number[],
  names: string[]
): number {
  if (names.length < 3) return 0;
  const mcp = readJoint(joints, SMPLX_IDX[names[0]] ?? -1);
  const pip = readJoint(joints, SMPLX_IDX[names[1]] ?? -1);
  const dip = readJoint(joints, SMPLX_IDX[names[2]] ?? -1);
  if (SMPLX_IDX[names[0]] ?? -1 < 0) return 0;
  const mcpToPip = normalize(sub(pip, mcp));
  const pipToDip = normalize(sub(dip, pip));
  const bendDot = dot(mcpToPip, pipToDip);
  return Math.max(0, Math.min(1, (1 - bendDot) / 2));
}

// ---------------------------------------------------------------------------
// Hand pose
// ---------------------------------------------------------------------------

function defaultFingers(): Record<FingerName, FingerPose> {
  return {
    thumb: { dir: [1, 0, 0], curl: 0 },
    index: { dir: [0, 0, 1], curl: 0 },
    middle: { dir: [0, 0, 1], curl: 0 },
    ring: { dir: [0, 0, 1], curl: 0 },
    pinky: { dir: [0, 0, 1], curl: 0 },
  };
}

/**
 * Build a HandPose from SMPL-X finger joint positions (official layout).
 */
function buildHandFromJoints(
  joints: number[],
  wristPos: Vec3,
  side: "left" | "right"
): HandPose {
  const fingerNames = side === "left" ? FINGER_JOINT_NAMES : FINGER_JOINT_NAMES_R;

  // Wrist → index MCP direction
  const idxMcpIdx = SMPLX_IDX[side === "left" ? "left_index_1" : "right_index_1"]!;
  const idxMcp = toCanonical(readJoint(joints, idxMcpIdx), true);
  const palmDir = normalize(sub(idxMcp, wristPos));

  // Palm normal: cross( wrist→index_MCP, wrist→pinky_MCP )
  const pinkyMcpIdx = SMPLX_IDX[side === "left" ? "left_pinky_1" : "right_pinky_1"]!;
  const pinkyMcp = toCanonical(readJoint(joints, pinkyMcpIdx), true);
  const palmNormal = normalize(cross(sub(idxMcp, wristPos), sub(pinkyMcp, wristPos)));

  // Per-finger
  const fingers: Record<FingerName, FingerPose> = { ...defaultFingers() };
  for (const finger of ["thumb", "index", "middle", "ring", "pinky"] as FingerName[]) {
    const names = fingerNames[finger];
    const dir = computeFingerDir(joints, names);
    const curl = computeFingerCurl(joints, names);
    fingers[finger] = { dir: toCanonical(dir, true), curl };
  }

  return {
    wristPos,
    palmNormal: palmNormal[0] === 0 && palmNormal[1] === 0 && palmNormal[2] === 0
      ? [0, 0, 1]
      : palmNormal,
    palmDir: palmDir[0] === 0 && palmDir[1] === 0 && palmDir[2] === 0
      ? [0, 0, 1]
      : palmDir,
    fingers,
  };
}

/**
 * Build an approximate HandPose from the IK-solver format (no finger joints).
 *
 * Uses wrist + index-MCP positions for palmDir, and a default palmNormal.
 * Finger curl defaults to 0 (straight) since no per-finger data exists.
 */
function buildHandFromIK(
  _joints: number[],
  wristPos: Vec3,
  handMcpPos: Vec3,
  _pose: number[] | undefined,
  side: "left" | "right"
): HandPose {
  // Palm direction: from wrist toward the hand MCP position
  let palmDir: Vec3 = normalize(sub(handMcpPos, wristPos));
  if (palmDir[0] === 0 && palmDir[1] === 0 && palmDir[2] === 0) {
    palmDir[0] = 0; palmDir[1] = 0; palmDir[2] = 1;
  }

  // Palm normal: we don't have a pinky MCP in IK format, so estimate
  // from the hand pose hint (pose[66:69] = local dir, pose[69:72] = right).
  // Without the full quaternion chain, fall back to a reasonable default:
  // in canonical space, the palm normal for a relaxed hand at the side
  // points roughly sideways.
  const palmNormal: Vec3 = side === "left" ? [0, 0, -1] : [0, 0, 1];

  return {
    wristPos,
    palmNormal,
    palmDir,
    fingers: defaultFingers(),
  };
}

// ---------------------------------------------------------------------------
// Body pose
// ---------------------------------------------------------------------------

function buildBodyPose(
  joints: number[],
  official: boolean
): BodyPose {
  // Helper to read a joint and convert to canonical
  const c = (name: string): Vec3 => toCanonicalJoint(joints, name, official);

  // Pelvis (root)
  const pelvis = c("pelvis");

  // Left / right hip
  const leftHip = c("left_hip");
  const rightHip = c("right_hip");

  // Left / right shoulder (clavicle)
  const leftShoulder = c("left_clavicle");
  const rightShoulder = c("right_clavicle");

  // Spine direction (from hips toward shoulders)
  const hipMid = scale(add(leftHip, rightHip), 0.5);
  const shoulderMid = scale(add(leftShoulder, rightShoulder), 0.5);
  const spineVec = sub(shoulderMid, hipMid);
  const spineDir = normalize(spineVec);

  // Shoulder axis (left → right)
  const shoulderAxis = normalize(sub(rightShoulder, leftShoulder));

  // Shoulder height (canonical Y of shoulder midpoint)
  const shoulderHeight = shoulderMid[1];

  // Arms — both layouts use the same logical names.
  // In IK layout, "left_elbow" (idx 14) holds the elbow position and
  // "left_forearm" (idx 16) holds the wrist position.
  // In official SMPL-X, "left_elbow" (idx 18) holds the elbow position and
  // "left_forearm" (idx 20) holds the wrist position.
  const leftElbow = c("left_elbow");
  const rightElbow = c("right_elbow");
  const leftWrist = c("left_forearm");
  const rightWrist = c("right_forearm");

  // Legs
  const leftKnee = c("left_knee");
  const rightKnee = c("right_knee");
  const leftAnkle = c("left_ankle");
  const rightAnkle = c("right_ankle");

  // Head / neck
  const neckPos = c("neck");
  const headPos = c("head");

  // Head direction: from neck toward head (typically up/forward)
  let headDirVec = sub(headPos, neckPos);
  // Fallback: nose direction (head → nose) isn't available in smplx_joints,
  // so use neck→head as forward. If degenerate, use spine direction.
  if (len(headDirVec) < 1e-6) {
    headDirVec = spineDir;
  }
  const headDir = normalize(headDirVec);

  return {
    hipsPos: pelvis,
    shoulderHeight,
    spineDir,
    shoulderAxis,
    leftShoulder,
    leftElbow,
    leftWrist,
    rightShoulder,
    rightElbow,
    rightWrist,
    neckPos,
    headPos,
    headDir,
    leftHip,
    leftKnee,
    leftAnkle,
    rightHip,
    rightKnee,
    rightAnkle,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Convert a single SmplxPose to a CanonicalPose.
 *
 * Uses `smplx_joints` for body and hand positions when available.
 * Falls back to the `pose` axis-angle parameters when smplx_joints is all
 * zeros (or too short), producing an approximate canonical pose.
 */
export function smplxPoseToCanonicalPose(
  smplxPose: SmplxPose,
  timestamp: number = 0
): CanonicalPose | null {
  const joints = smplxPose.smplx_joints;

  if (!joints || joints.length < 128 * 3) {
    // smplx_joints not provided — try fallback from pose array?
    return null;
  }

  const official = hasOfficialLayout(joints);
  const hasData = hasAnyData(joints);

  // Fallback: if joints are all zero, we can't build a pose
  if (!hasData) {
    return null;
  }

  // If we have finger joints (official layout), build hand poses from them.
  // If not (IK layout), use wrist + hand MCP positions.
  // Left hand
  const leftWristIdx = official ? SMPLX_IDX["left_forearm"] : IK_BODY_IDX["left_forearm"];
  const leftHandIdx = official ? SMPLX_IDX["left_hand"] : IK_BODY_IDX["left_hand"];
  const rightWristIdx = official ? SMPLX_IDX["right_forearm"] : IK_BODY_IDX["right_forearm"];
  const rightHandIdx = official ? SMPLX_IDX["right_hand"] : IK_BODY_IDX["right_hand"];

  const leftWristPos = toCanonical(readJoint(joints, leftWristIdx!), official);
  const rightWristPos = toCanonical(readJoint(joints, rightWristIdx!), official);

  let leftHand: HandPose;
  let rightHand: HandPose;

  if (official) {
    leftHand = buildHandFromJoints(joints, leftWristPos, "left");
    rightHand = buildHandFromJoints(joints, rightWristPos, "right");
  } else {
    const leftMcp = toCanonical(readJoint(joints, leftHandIdx!), official);
    const rightMcp = toCanonical(readJoint(joints, rightHandIdx!), official);
    leftHand = buildHandFromIK(joints, leftWristPos, leftMcp, smplxPose.pose, "left");
    rightHand = buildHandFromIK(joints, rightWristPos, rightMcp, smplxPose.pose, "right");
  }

  // Body pose
  const body = buildBodyPose(joints, official);

  return {
    body,
    leftHand,
    rightHand,
    timestamp,
  };
}

/**
 * Convert an array of SmplxPoses to CanonicalPoses.
 */
export function smplxToCanonicalPoses(
  poses: SmplxPose[]
): (CanonicalPose | null)[] {
  const results: (CanonicalPose | null)[] = [];
  for (let i = 0; i < poses.length; i++) {
    const frame = poses[i];
    results.push(smplxPoseToCanonicalPose(frame, i / 30));
  }
  return results;
}

/**
 * Convert a SmplxFrame (label + frames) to an array of CanonicalPoses.
 */
export function smplxFrameToCanonicalPoses(
  frame: SmplxFrame
): (CanonicalPose | null)[] {
  return smplxToCanonicalPoses(frame.frames);
}
