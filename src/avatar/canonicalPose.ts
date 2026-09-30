/**
 * Unified coordinate system and canonical pose definition.
 *
 * Convention (X-forward, Y-up, Z-right, right-handed, meters):
 * - X = forward (out of the screen / toward the viewer)
 * - Y = up
 * - Z = right
 *
 * This is the single canonical space the entire retargeting pipeline
 * reads from and writes to. Source adapters (MediaPipe, SMPL-X) and the
 * retargeter all convert to/from this space.
 */
import { Vector3, Quaternion, Matrix4 } from "three";

export const CANONICAL_UP = new Vector3(0, 1, 0);
export const CANONICAL_FORWARD = new Vector3(1, 0, 0);
export const CANONICAL_RIGHT = new Vector3(0, 0, 1);

export type CanonicalCoord = {
  x: number;
  y: number;
  z: number;
};

export type Vec3 = [number, number, number];

export type Side = "left" | "right";
export type Limb = "arm" | "leg";

/** Per-bone local quaternion (relative to parent in the avatar's local space). */
export type BoneRotation = Record<string, Quaternion>;

/** Per-bone world position of the joint (used for IK solvers / debug). */
export type JointPositions = Record<string, Vector3>;

/**
 * Articulated finger definition.
 * `dir` is the world-space direction of the finger's *metacarpal* bone,
 * normalized to unit length. `curl` (0…1) is the flexion of the
 * proximal interphalangeal joint — 0 = straight, 1 = fully curled.
 */
export interface FingerPose {
  dir: Vec3;
  curl: number;
}

/** Which finger. Thumb is special-cased because it has a different bend plane. */
export type FingerName =
  | "thumb"
  | "index"
  | "middle"
  | "ring"
  | "pinky";

/** A hand with 5 articulated fingers. */
export interface HandPose {
  /** Wrist position in canonical space (meters). */
  wristPos: Vec3;
  /** Direction from wrist to index MCP, normalized. */
  palmNormal: Vec3;
  /** Direction from wrist to middle MCP, normalized. */
  palmDir: Vec3;
  /** Per-finger metadata. */
  fingers: Record<FingerName, FingerPose>;
}

/** Root / body joints tracked by the canonical pose. */
export interface BodyPose {
  /** Root (hips) world position in canonical space. */
  hipsPos: Vec3;
  /** Shoulder height (average of left/right shoulder Y). */
  shoulderHeight: number;
  /** Direction from hips to shoulders (normalized). */
  spineDir: Vec3;
  /** Direction across the shoulders L→R. */
  shoulderAxis: Vec3;
  /** Direction from shoulder to elbow to wrist (elbow position). */
  leftShoulder: Vec3;
  leftElbow: Vec3;
  leftWrist: Vec3;
  rightShoulder: Vec3;
  rightElbow: Vec3;
  rightWrist: Vec3;
  /** Head / neck. */
  neckPos: Vec3;
  headPos: Vec3;
  headDir: Vec3;
  /** Hip → toe direction (used for leg retargeting). */
  leftHip: Vec3;
  leftKnee: Vec3;
  leftAnkle: Vec3;
  rightHip: Vec3;
  rightKnee: Vec3;
  rightAnkle: Vec3;
}

/**
 * The full canonical pose — the intermediate representation that all
 * source adapters produce and the retargeter consumes.
 */
export interface CanonicalPose {
  body: BodyPose;
  leftHand: HandPose;
  rightHand: HandPose;
  /** Timestamp in seconds (source frame time). */
  timestamp: number;
}

export interface CanonicalPoseFrame {
  pose: CanonicalPose;
  deltaTime: number;
}

/**
 * Convert a Vec3 (tuple) to a Three.js Vector3. No allocation of the vector
 * itself if `out` is provided.
 */
export function toVec3(v: Vec3, out = new Vector3()): Vector3 {
  out.set(v[0], v[1], v[2]);
  return out;
}

export function fromVec3(v: Vector3): Vec3 {
  return [v.x, v.y, v.z];
}

/**
 * PHASE 11 — scratch for `rotationBetween`.
 *
 * This is the hottest primitive in the pipeline: it is called once per bone per
 * retarget (~134 times for a full skeleton with hands), and each call
 * allocated 1-2 Vector3s. It is a LEAF function — it calls only in-place
 * Three.js methods (`crossVectors`, `normalize`, `set`) and never re-enters
 * this module — so these two singletons cannot be clobbered by a nested call.
 *
 * They are module-scoped rather than arena-allocated because `rotationBetween`
 * is re-entrant across callers (bodyRetargeter and handRetargeter both call it
 * while holding their own scratch), and a leaf with no callee of its own is
 * exactly the safe case for module scratch. The returned quaternion is the
 * caller's `out`, so nothing here escapes.
 */
const _rbCross = new Vector3();
const _rbOrtho = new Vector3();

/**
 * Build a quaternion that rotates `from` direction to `to` direction using
 * the shortest arc (Hamilton / Three.js convention). Handles parallel and
 * antiparallel vectors.
 */
export function rotationBetween(
  from: Vector3,
  to: Vector3,
  out = new Quaternion()
): Quaternion {
  const lenProduct = Math.sqrt(from.lengthSq() * to.lengthSq());
  if (lenProduct < 1e-12) {
    return out.identity();
  }
  // `out` must be the shortest-arc rotation taking `from` onto `to`.  For vectors
  // of length |a|,|b| separated by theta, that is
  //     q = normalize( (a x b) , |a||b| + a.b ) = normalize( (sin t * axis) , (1 + cos t) )
  // The `w` term MUST be |a||b| + dot.  (An earlier form using
  // `dot + sqrt(|a|^2|b|^2 + dot^2)` is only correct at 0deg / 90deg / 180deg:
  // it under-rotates by ~17% for small angles and collapses a 170deg aim to 45deg.)
  // PHASE 11: was `new Vector3()`. Same operation, same operand order.
  const c = _rbCross.crossVectors(from, to);
  const w = lenProduct + from.dot(to);
  if (w < 1e-12 * lenProduct) {
    // 180deg flip — pick any axis orthogonal to `from` (a x b is degenerate here).
    const ortho = _rbOrtho.crossVectors(CANONICAL_UP, from);
    if (ortho.lengthSq() < 1e-12) {
      ortho.crossVectors(CANONICAL_FORWARD, from);
    }
    ortho.normalize();
    return out.set(ortho.x, ortho.y, ortho.z, 0).normalize();
  }
  out.set(c.x, c.y, c.z, w);
  return out.normalize();
}

/** Result of decomposing a matrix. */
export interface DecomposeResult {
  position: Vector3;
  quaternion: Quaternion;
  scale: Vector3;
}

/**
 * Decompose a 4×4 matrix into position / quaternion / scale.
 */
export function decomposeMatrix(m: Matrix4): DecomposeResult {
  const position = new Vector3();
  const quaternion = new Quaternion();
  const scale = new Vector3();
  m.decompose(position, quaternion, scale);
  return { position, quaternion, scale };
}

/**
 * Normalize a direction vector. If it's near-zero, returns the `fallback`.
 */
export function safeNormalize(v: Vector3, fallback: Vector3): Vector3 {
  if (v.lengthSq() < 1e-12) {
    return fallback.clone();
  }
  return v.clone().normalize();
}
