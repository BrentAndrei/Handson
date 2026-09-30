/**
 * Coordinate system conversion utilities.
 *
 * Canonical space:  X-forward, Y-up, Z-right  (right-handed, meters)
 *
 * Sources:
 *  - MediaPipe:  X-right, Y-down, Z-inward (right-handed)
 *  - SMPL-X:     X-right, Y-up, Z-back   (right-handed, OpenCV convention)
 *  - Three.js / GLTF:  X-right, Y-up, Z-outward (right-handed, WebGL)
 *
 * All adapters convert to canonical first, then the retargeter converts
 * canonical → avatar-local.
 */
import { Vector3, Matrix4 } from "three";
import { Vec3 } from "./canonicalPose";

export type Coords = Vec3;

/**
 * MediaPipe landmark → canonical (X-fwd, Y-up, Z-right = SUBJECT'S right).
 *
 * Input is a RAW, unmirrored camera frame (the app feeds ctx.drawImage(video)
 * straight into detectForVideo — no flip anywhere), and the signer FACES the
 * camera. For a camera-facing signer the anatomical left is on the image's
 * RIGHT half, so:
 *   fwd   = -Zmp  (MediaPipe +z is away from camera; forward = toward it)
 *   up    = -Ymp  (image y grows downward)
 *   right = -Xmp  (subject's right hand is on the image's LEFT)
 * which lands canonical +X/+Y/+Z on the subject's forward/up/right.
 *
 * E8 history: this previously returned [-v[2], -v[1], v[0]] — mapping
 * image-right to canonical +Z. For a camera-facing signer that is a
 * REFLECTION (determinant -1) which swaps left/right (subject's left landed
 * on +Z). Verified numerically by scripts/verify-mediapipe-handedness.ts.
 * If a selfie-mirrored input mode is ever introduced, this sign must flip.
 */
export function mediapipeToCanonical(v: Coords): Coords {
  return [-v[2], -v[1], -v[0]];
}

/**
 * Convert a full Float32Array of MediaPipe landmarks (interleaved xyz)
 * to canonical Coords array. Reuses `out` arrays when provided.
 */
export function mediapipeArrayToCanonical(
  raw: number[] | Float32Array,
  out: Vector3[] = []
): Vector3[] {
  const n = (raw.length / 3) | 0;
  for (let i = 0; i < n; i++) {
    let v = out[i];
    if (!v) {
      v = new Vector3();
      out[i] = v;
    }
    const x = raw[i * 3];
    const y = raw[i * 3 + 1];
    const z = raw[i * 3 + 2];
    // MediaPipe → canonical (same map as mediapipeToCanonical; E8 fix: -x)
    v.set(-z, -y, -x);
  }
  return out;
}

/** SMPL-X (OpenCV: X-right, Y-up, Z-back) → canonical (X-fwd, Y-up, Z-right). */
export function smplxToCanonical(v: Coords): Coords {
  // SMPL-X: X-right, Y-up, Z-back
  // Canonical: X-forward (toward viewer), Y-up, Z-right
  //   forward = -Z
  //   up      = Y
  //   right   = X
  // Mirror flip (GLTF -X ↔ canonical +Z) is handled in canonicalToGltf.
  return [-v[2], v[1], v[0]];
}

/** GLTF / Three.js world → canonical. GLTF: X-right, Y-up, Z-out. */
export function gltfToCanonical(v: Coords): Coords {
  // GLTF: X-right, Y-up, Z-outward (toward viewer)
  // Canonical: X-forward (toward viewer), Y-up, Z-right
  //   forward = Z (outward → forward)
  //   up      = Y
  //   right   = -X  (mirrored avatar: GLTF -X is avatar's right = canonical right)
  return [v[2], v[1], -v[0]];
}

/** Canonical → GLTF world. */
export function canonicalToGltf(v: Coords): Coords {
  // Mirror: canonical +Z (right) → GLTF -X (avatar's right)
  return [-v[2], v[1], v[0]];
}

/**
 * Scale a canonical vector set by a uniform factor (e.g. GLB armature
 * scale 0.01 = cm→m). Returns a new array.
 */
export function scaleCoords(v: Coords, s: number): Coords {
  return [v[0] * s, v[1] * s, v[2] * s];
}

/**
 * Build a rotation matrix that maps the source basis vectors to the
 * canonical basis. Useful for debugging coordinate transforms.
 */
export function basisMatrix(
  right: Vector3,
  up: Vector3,
  forward: Vector3
): Matrix4 {
  const m = new Matrix4();
  m.makeBasis(right, up, forward);
  return m;
}
