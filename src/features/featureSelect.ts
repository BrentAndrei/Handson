import {
  POSE_COUNT,
  POSE_OFFSET,
  HAND_COUNT,
  LEFT_HAND_OFFSET,
  RIGHT_HAND_OFFSET,
  FACE_OFFSET,
} from "../mediapipe/types";

/**
 * Curated subset of MediaPipe FaceMesh's 468 points covering the outer +
 * inner lip contour (mirrors MediaPipe's own FACEMESH_LIPS connection set).
 * Mouth shape carries grammatical/non-manual information in FSL (mouthed
 * cues, negation, intensity), so we keep it — but including all 468 face
 * points would dominate the model's per-frame input width for very little
 * added signal on a hand-collected dataset, so everything else about the
 * face is dropped here.
 */
export const LIP_LANDMARK_INDICES = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
  14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
  312, 13, 82, 81, 42, 183, 78,
];

// pose(33) + leftHand(21) + rightHand(21) + lips(40) points, x/y/z each.
export const FEATURE_DIM =
  (POSE_COUNT + HAND_COUNT * 2 + LIP_LANDMARK_INDICES.length) * 3;

/**
 * Reduces a full normalized holistic vector (as produced by
 * normalizeLandmarks) down to FEATURE_DIM floats: pose + both hands + the
 * lip subset above. Run this AFTER normalizeLandmarks() — normalization
 * still computes its shoulder-anchor/scale over the full packed layout, so
 * that math is untouched; this only decides which already-normalized
 * points get fed to the sequence model. Keeping this as a separate step
 * (rather than baking it into normalizeLandmarks) means the overlay
 * drawing and any other consumer of the full vector are unaffected.
 */
export function selectFeatures(vector: Float32Array): Float32Array {
  const out = new Float32Array(FEATURE_DIM);
  let o = 0;
  o = copyRange(vector, POSE_OFFSET, POSE_COUNT, out, o);
  o = copyRange(vector, LEFT_HAND_OFFSET, HAND_COUNT, out, o);
  o = copyRange(vector, RIGHT_HAND_OFFSET, HAND_COUNT, out, o);
  o = copyIndices(vector, FACE_OFFSET, LIP_LANDMARK_INDICES, out, o);
  return out;
}

function copyRange(
  src: Float32Array,
  offset: number,
  count: number,
  out: Float32Array,
  o: number
): number {
  out.set(src.subarray(offset, offset + count * 3), o);
  return o + count * 3;
}

function copyIndices(
  src: Float32Array,
  groupOffset: number,
  indices: number[],
  out: Float32Array,
  o: number
): number {
  for (const idx of indices) {
    const base = groupOffset + idx * 3;
    out[o++] = src[base];
    out[o++] = src[base + 1];
    out[o++] = src[base + 2];
  }
  return o;
}
