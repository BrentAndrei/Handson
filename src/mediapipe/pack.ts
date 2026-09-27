import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
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
  type NormalizedLandmark,
} from "./types";

function writeLandmarks(
  target: Float32Array,
  offset: number,
  count: number,
  landmarks: NormalizedLandmark[] | undefined
): boolean {
  if (!landmarks || landmarks.length === 0) {
    target.fill(0, offset, offset + count * 3);
    return false;
  }
  const n = Math.min(count, landmarks.length);
  for (let i = 0; i < n; i++) {
    const p = landmarks[i];
    const base = offset + i * 3;
    target[base] = p?.x ?? 0;
    target[base + 1] = p?.y ?? 0;
    target[base + 2] = p?.z ?? 0;
  }
  if (n < count) {
    target.fill(0, offset + n * 3, offset + count * 3);
  }
  return true;
}

/** Packs a raw HolisticLandmarkerResult into the shared flat buffer layout. */
export function packHolisticResult(
  result: HolisticLandmarkerResult,
  timestamp: number
): LandmarksResult {
  const buffer = new ArrayBuffer(TOTAL_FLOATS * 4);
  const view = new Float32Array(buffer);

  // Each field is one entry per detected instance (NormalizedLandmark[][]);
  // HolisticLandmarker only ever tracks a single person, so the landmark
  // set we want is always index [0] — absent entirely when nothing of
  // that type was detected in this frame.
  const hasPose = writeLandmarks(view, POSE_OFFSET, POSE_COUNT, result.poseLandmarks[0]);
  const hasLeftHand = writeLandmarks(
    view,
    LEFT_HAND_OFFSET,
    HAND_COUNT,
    result.leftHandLandmarks[0]
  );
  const hasRightHand = writeLandmarks(
    view,
    RIGHT_HAND_OFFSET,
    HAND_COUNT,
    result.rightHandLandmarks[0]
  );
  const hasFace = writeLandmarks(view, FACE_OFFSET, FACE_COUNT, result.faceLandmarks[0]);

  return { timestamp, buffer, hasPose, hasLeftHand, hasRightHand, hasFace };
}
