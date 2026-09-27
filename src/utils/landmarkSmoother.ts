import { POSE_COUNT, HAND_COUNT, FACE_COUNT, POSE_OFFSET, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET, FACE_OFFSET } from "../mediapipe/types";

/**
 * Moving-average landmark smoother applied to raw landmark buffers
 * BEFORE normalization and feature extraction, so the inference
 * pipeline receives temporally stable inputs. Separate from the
 * display-only One Euro Filter in smoothing.ts.
 *
 * Window of 3 frames kills high-frequency jitter without introducing
 * perceptible lag (at 24fps, 3 frames = ~125ms).
 */
export class LandmarkSmoother {
  private readonly windowSize: number;
  private readonly buffers: Map<number, Float32Array> = new Map();
  private index = 0;
  private count = 0;

  constructor(windowSize = 3) {
    this.windowSize = windowSize;
  }

  /**
   * Smooth a raw landmark buffer in-place. The buffer contains
   * all landmark groups packed sequentially per the Hand-Skeleton
   * packed layout (see mediapipe/types.ts).
   */
  smooth(buffer: Float32Array): Float32Array {
    if (this.count >= this.windowSize) {
      this.buffers.clear();
      this.index = 0;
      this.count = 0;
    }

    const result = new Float32Array(buffer.length);
    result.set(buffer);

    this.buffers.set(this.index, buffer.slice());
    this.index = (this.index + 1) % this.windowSize;
    this.count++;

    const valid = Math.min(this.count, this.windowSize);

    const groups = [
      { offset: POSE_OFFSET, count: POSE_COUNT },
      { offset: LEFT_HAND_OFFSET, count: HAND_COUNT },
      { offset: RIGHT_HAND_OFFSET, count: HAND_COUNT },
      { offset: FACE_OFFSET, count: FACE_COUNT },
    ];

    for (const { offset, count } of groups) {
      for (let i = 0; i < count; i++) {
        const base = (offset + i) * 3;
        let sx = 0, sy = 0, sz = 0;
        for (let w = 0; w < valid; w++) {
          const ring = this.buffers.get(w)!;
          sx += ring[base];
          sy += ring[base + 1];
          sz += ring[base + 2];
        }
        result[base] = sx / valid;
        result[base + 1] = sy / valid;
        result[base + 2] = sz / valid;
      }
    }

    return result;
  }

  reset(): void {
    this.buffers.clear();
    this.index = 0;
    this.count = 0;
  }
}
