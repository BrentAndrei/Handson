import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export type { NormalizedLandmark };

// Packed layout, all in a single Float32Array-backed ArrayBuffer:
// [0      .. 99)   pose:       33 points * 3 (x,y,z)
// [99     .. 162)  leftHand:   21 points * 3
// [162    .. 225)  rightHand:  21 points * 3
// [225    .. 1659) face:       478 points * 3
export const POSE_COUNT = 33;
export const HAND_COUNT = 21;
export const FACE_COUNT = 468;
export const POSE_OFFSET = 0;
export const LEFT_HAND_OFFSET = POSE_OFFSET + POSE_COUNT * 3;
export const RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + HAND_COUNT * 3;
export const FACE_OFFSET = RIGHT_HAND_OFFSET + HAND_COUNT * 3;
export const TOTAL_FLOATS = FACE_OFFSET + FACE_COUNT * 3;

export interface LandmarksResult {
  timestamp: number;
  buffer: ArrayBuffer;
  hasPose: boolean;
  hasLeftHand: boolean;
  hasRightHand: boolean;
  hasFace: boolean;
}
