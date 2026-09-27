import {
  FACE_COUNT,
  FACE_OFFSET,
  HAND_COUNT,
  LEFT_HAND_OFFSET,
  POSE_COUNT,
  POSE_OFFSET,
  RIGHT_HAND_OFFSET,
  TOTAL_FLOATS,
  type LandmarksResult,
} from "../mediapipe/types";

// Mirrors the packed layout written in mediapipe/pack.ts

// MediaPipe Pose landmark indices
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

export interface NormalizedFrame {
  /** Same packed layout as the raw buffer, but translated + scaled in place. */
  vector: Float32Array;
  anchor: [number, number, number];
  scale: number;
  /** True if pose wasn't confidently detected and we fell back to a
   *  full-vector bounding-box scale instead of shoulder-width. Useful for
   *  downstream confidence weighting. */
  usedFallbackScale: boolean;
}

/**
 * Translates + scales a packed landmark buffer in place.
 *
 * Anchor: midpoint of the shoulders. Stable across most sign-language framing
 * (upper body, seated or standing) and unaffected by hand motion, unlike
 * using a hand or face point as the anchor.
 *
 * Scale: shoulder-to-shoulder distance. Chosen over a full bounding-box
 * diagonal because it's a fixed anatomical reference — it doesn't change
 * when the signer's hands extend outward mid-sign, which a bbox-diagonal
 * scale would (causing every OTHER point to visually "shrink" as the hands
 * move, i.e. scale leaking motion information into what should be a
 * pose-invariant channel).
 *
 * Falls back to a bounding-box diagonal of whatever landmarks ARE present
 * only when pose landmarks are missing entirely (e.g. signer stepped out of
 * frame from the waist up but a hand is still visible) — this fallback
 * exists to prevent divide-by-zero / NaN propagation, not as a preferred path.
 */
export function normalizeLandmarks(msg: LandmarksResult): NormalizedFrame {
  const source = new Float32Array(msg.buffer);
  const vector = new Float32Array(TOTAL_FLOATS);

  let anchor: [number, number, number];
  let scale: number;
  let usedFallbackScale = false;

  if (msg.hasPose) {
    anchor = midpoint(source, POSE_OFFSET, LEFT_SHOULDER, RIGHT_SHOULDER);
    const shoulderDist = distance(
      source,
      POSE_OFFSET,
      LEFT_SHOULDER,
      RIGHT_SHOULDER
    );

    // Guard against a degenerate near-zero shoulder distance (e.g. a
    // momentary bad pose estimate) rather than dividing by ~0.
    if (shoulderDist > 1e-4) {
      scale = shoulderDist;
    } else {
      scale = boundingBoxDiagonal(source, 0, TOTAL_FLOATS / 3);
      usedFallbackScale = true;
    }
  } else {
    // No pose at all: fall back to hip-independent bbox of whatever
    // landmark groups ARE present, anchored on their centroid.
    anchor = centroid(source, msg);
    scale = boundingBoxDiagonal(source, 0, TOTAL_FLOATS / 3);
    usedFallbackScale = true;
  }

  if (scale <= 1e-4) scale = 1; // final safety net — never divide by ~0

  for (let i = 0; i < TOTAL_FLOATS; i += 3) {
    vector[i] = (source[i] - anchor[0]) / scale;
    vector[i + 1] = (source[i + 1] - anchor[1]) / scale;
    vector[i + 2] = (source[i + 2] - anchor[2]) / scale;
  }

  return { vector, anchor, scale, usedFallbackScale };
}

function pointAt(
  buf: Float32Array,
  groupOffset: number,
  index: number
): [number, number, number] {
  const base = groupOffset + index * 3;
  return [buf[base], buf[base + 1], buf[base + 2]];
}

function midpoint(
  buf: Float32Array,
  groupOffset: number,
  a: number,
  b: number
): [number, number, number] {
  const pa = pointAt(buf, groupOffset, a);
  const pb = pointAt(buf, groupOffset, b);
  return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2];
}

function distance(
  buf: Float32Array,
  groupOffset: number,
  a: number,
  b: number
): number {
  const pa = pointAt(buf, groupOffset, a);
  const pb = pointAt(buf, groupOffset, b);
  return Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]);
}

function centroid(
  buf: Float32Array,
  msg: LandmarksResult
): [number, number, number] {
  let sx = 0,
    sy = 0,
    sz = 0,
    n = 0;
  const groups: [boolean, number, number][] = [
    [msg.hasPose, POSE_OFFSET, POSE_COUNT],
    [msg.hasLeftHand, LEFT_HAND_OFFSET, HAND_COUNT],
    [msg.hasRightHand, RIGHT_HAND_OFFSET, HAND_COUNT],
    [msg.hasFace, FACE_OFFSET, FACE_COUNT],
  ];
  for (const [present, offset, count] of groups) {
    if (!present) continue;
    for (let i = 0; i < count; i++) {
      const base = offset + i * 3;
      sx += buf[base];
      sy += buf[base + 1];
      sz += buf[base + 2];
      n++;
    }
  }
  if (n === 0) return [0, 0, 0];
  return [sx / n, sy / n, sz / n];
}

function boundingBoxDiagonal(
  buf: Float32Array,
  startPoint: number,
  endPoint: number
): number {
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;

  for (let p = startPoint; p < endPoint; p++) {
    const base = p * 3;
    const x = buf[base],
      y = buf[base + 1],
      z = buf[base + 2];
    // Skip zero-filled (absent) landmark groups so they don't collapse
    // the bbox toward the origin.
    if (x === 0 && y === 0 && z === 0) continue;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  if (!isFinite(minX)) return 1; // nothing detected at all
  return Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
}

/** Unused hip constants kept for callers who want a hip-based anchor variant
 *  (e.g. lower-body-inclusive sign systems). Left exported for extension. */
export const LANDMARK_INDICES = {
  LEFT_SHOULDER,
  RIGHT_SHOULDER,
  LEFT_HIP,
  RIGHT_HIP,
};
