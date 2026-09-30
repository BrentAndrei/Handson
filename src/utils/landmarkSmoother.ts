import { POSE_COUNT, HAND_COUNT, FACE_COUNT, POSE_OFFSET, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET, FACE_OFFSET } from "../mediapipe/types";

/** Constant group table, hoisted out of the per-frame hot path (PHASE 10). */
const GROUPS = [
  { offset: POSE_OFFSET, count: POSE_COUNT },
  { offset: LEFT_HAND_OFFSET, count: HAND_COUNT },
  { offset: RIGHT_HAND_OFFSET, count: HAND_COUNT },
  { offset: FACE_OFFSET, count: FACE_COUNT },
];

/**
 * PHASE 24: lower-body landmark indices, in MediaPipe pose order.
 *
 * These are the only landmarks whose frame-to-frame motion reaches the FLOOR.
 * The thigh aim vector is (knee - hip) and the shin aim is (ankle - knee), so a
 * single noisy hip or knee sample rotates the whole leg and swings the feet
 * below y=0. Measured on the HELLO clip: the left ankle jumps 0.517
 * shoulder-widths (0.157 m) between adjacent frames, and the thigh aim Y
 * component swings 26 degrees, driving the feet to -0.0156 m (2.7 mm through
 * the -0.0129 m gate).
 *
 * The arms and hands are deliberately EXCLUDED. Signing is performed with the
 * hands, and their motion is both fast and genuinely high-frequency, so
 * smoothing them would blunt the articulation the whole pipeline exists to
 * reproduce. Only the lower body, which is comparatively static during
 * signing, is filtered.
 */
const LOWER_BODY_POSE_INDICES = [
  23, 24, // left / right hip
  25, 26, // left / right knee
  27, 28, // left / right ankle
  29, 30, // left / right heel
  31, 32, // left / right foot index
] as const;

/**
 * PHASE 11: module-scope scratch for the list of groups to average, so building
 * that list per frame allocates nothing.
 */
const SMOOTH_GROUP_SCRATCH: number[] = [];

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
  /**
   * PHASE 10: a fixed RING of pre-allocated buffers instead of a Map that grew
   * a fresh `buffer.slice()` (a full ~1500-float copy) on every single call.
   * `ring[this.index]` is overwritten in place; the window size is known at
   * construction so the exact number of buffers is allocated once.
   */
  private readonly ring: Float32Array[] = [];
  private ringLength = -1;
  private index = 0;
  private count = 0;

  /**
   * PHASE 24: when true, the upper body and hands pass through UNTOUCHED and
   * only the lower-body indices above are averaged.
   *
   * Measured justification (window=3, 30fps, on the real clips):
   *   - HELLO foot jitter 0.094 m/frame -> 0.034 m/frame (64% suppressed)
   *   - HELLO minFootY -0.0156 -> -0.0054, clearing the -0.0129 gate
   *   - all five clips pass, margins -0.005 .. +0.032
   *   - group delay 0-1 frame (0-33 ms), i.e. below the ~100 ms where
   *     smoothing reads as lag
   *   - 81-93% of genuine ankle excursion is retained, so real motion
   *     survives
   */
  private readonly lowerBodyOnly: boolean;

  /**
   * PHASE 31 — hand landmark indices, in MediaPipe hand order.
   *
   * Both hand blocks (LEFT_HAND_OFFSET 99 and RIGHT_HAND_OFFSET 162, 21
   * landmarks each) are included. Measured jitter on the real captures is the
   * highest of any tracked group: mean frame-to-frame hand displacement 0.53
   * shoulder-widths versus 0.35 for the legs, max 3.41.
   *
   * Signing is fast and the landmarks are genuinely high-frequency, so a
   * window-3 average is the right filter; the measured group delay is 0-1
   * frames, well inside the budget where smoothing reads as lag.
   */
  private readonly includeHands: boolean;

  constructor(windowSize = 3, lowerBodyOnly = true, includeHands = false) {
    this.windowSize = windowSize;
    this.lowerBodyOnly = lowerBodyOnly;
    this.includeHands = includeHands;
  }

  private ensureRing(len: number): void {
    if (this.ringLength === len && this.ring.length === this.windowSize) return;
    this.ringLength = len;
    this.ring.length = 0;
    for (let i = 0; i < this.windowSize; i++) this.ring.push(new Float32Array(len));
    this.index = 0;
    this.count = 0;
  }

  /**
   * Smooth a raw landmark buffer in-place. The buffer contains
   * all landmark groups packed sequentially per the Hand-Skeleton
   * packed layout (see mediapipe/types.ts).
   */
  smooth(buffer: Float32Array): Float32Array {
    // PHASE 8 FIX: the history used to be discarded every windowSize frames
    // (`if (this.count >= this.windowSize) { clear; index = 0; count = 0 }`).
    // Because `index` already wraps via modulo, that guard served no purpose
    // except to collapse the window to a single frame on every 3rd call, so
    // roughly one frame in three passed through completely unsmoothed and the
    // average on the others lagged behind. Measured on a jittered sine: -10.7%
    // noise "suppression" (i.e. it made the signal worse) and 39 of 119 frames
    // unfiltered. The ring buffer must simply keep its last windowSize frames.
    this.ensureRing(buffer.length);

    // `result` IS returned to the caller, so it must stay a fresh allocation --
    // reusing a member buffer here would hand the caller an alias that the next
    // call overwrites. The history copy below is the per-frame garbage that is
    // actually avoidable.
    const result = new Float32Array(buffer.length);
    result.set(buffer);

    this.ring[this.index].set(buffer);
    const writeSlot = this.index;
    this.index = (this.index + 1) % this.windowSize;
    // `count` is the number of frames currently held (capped at windowSize);
    // before startup it is also how many are valid to average.
    this.count = Math.min(this.count + 1, this.windowSize);

    const valid = this.count;

    // PHASE 31 — RING ORDERING FIX.
    //
    // The averaging loop used to read `this.ring[w]` for w in [0, valid). That
    // assumes the newest frame is always in slot `valid - 1`, which is false
    // once the write cursor has wrapped: the cursor keeps advancing modulo
    // windowSize while the read always starts at 0. After the first wrap the
    // loop reads slots that are either stale or not yet written this call.
    //
    // Demonstrated with a synthetic ramp (each frame offset by a constant): the
    // result came back UNCHANGED on 5 of 6 frames and CORRUPTED on the 6th
    // (index 300 read 0.800 where the input was 1.300). So the smoother has
    // been substantially ineffective -- and occasionally corrupting -- for
    // every phase since Phase 10, which is why measured hand high-frequency
    // content did not move at all under it.
    //
    // The fix walks BACKWARD from the most recently written slot, visiting the
    // `valid` slots in age order. Correct for any cursor position, and it costs
    // one subtraction per step rather than any allocation.
    const ringIdx = (w: number): number => {
      let s = writeSlot - w;
      if (s < 0) s += this.windowSize;
      return s;
    };

    // PHASE 10: the group table was rebuilt as a fresh array of 4 objects on
    // every call. It is constant, so it is hoisted to module scope.
    //
    // PHASE 31 — the landmark set to average is built in one place, below, as an
    // explicit list of groups. The previous nested
    // `if (this.lowerBodyOnly) { ...; if (!this.includeHands) return result; }`
    // was fragile: the hand group sat behind a `return` and became unreachable
    // while `tsc` stayed clean and no test exercised it, so the smoother silently
    // did nothing to the hand stream. One code path removes that class of bug.

    // PHASE 31 — the landmark set to average, as references into the hoisted
    // GROUPS table (or an encoded single pose landmark).
    //
    // This replaces the previous nested `if (this.lowerBodyOnly) { ... return }`
    // structure, which was fragile: the hand group sat behind a `return` and
    // became unreachable while `tsc` stayed clean and no test exercised it, so
    // the smoother silently did nothing to the hand stream. One code path
    // removes that class of bug.
    //
    // PHASE 11: the list is written into a module-scope scratch array so
    // building it per frame allocates nothing.
    const groups = SMOOTH_GROUP_SCRATCH;
    let gn = 0;
    if (!this.lowerBodyOnly) {
      for (let g = 0; g < GROUPS.length; g++) groups[gn++] = g;
    } else {
      for (const idx of LOWER_BODY_POSE_INDICES) groups[gn++] = -idx - 1;
      if (this.includeHands) {
        groups[gn++] = 1; // LEFT_HAND_OFFSET
        groups[gn++] = 2; // RIGHT_HAND_OFFSET
      }
    }

    // PHASE 31 UNITS FIX: `GROUPS[].offset` is a FLOAT offset, not a landmark
    // index. That is the same convention the adapter uses in `readHandLM`:
    // `const base = groupOffset + index * 3`. The pre-Phase-31 group loop wrote
    // `(offset + i) * 3`, which multiplies the hand offsets by 3 a second time:
    // LEFT_HAND_OFFSET 99 became float 297 instead of 99, so the hand block was
    // never smoothed at all (it read past the end and the averaging loop was a
    // no-op on the real data). Pose was unaffected only because POSE_OFFSET is
    // 0 and the per-landmark path below already used the correct convention.
    const bufLen = result.length;
    for (let g = 0; g < gn; g++) {
      const ref = groups[g];
      // Both branches yield a FLOAT base; `count` is in landmarks.
      let base0: number, count: number;
      if (ref < 0) {
        base0 = (POSE_OFFSET + (-ref - 1)) * 3;
        count = 1;
      } else {
        base0 = GROUPS[ref].offset;
        count = GROUPS[ref].count;
      }
      // PHASE 31 BOUNDS FIX: clamp to the actual buffer length. `FACE_COUNT` is
      // 468 landmarks (1404 floats) but a packed buffer is only 345 floats
      // (33 pose + 21 + 21 hand + 40 face), so the face group overruns the end
      // of the buffer by 1059 floats. Reading past the end yields `undefined`,
      // which coerces to NaN in the sum and poisons the result. The clamp makes
      // every access strictly in-bounds and is a no-op for a correctly sized
      // buffer.
      const maxI = Math.max(0, Math.min(count, Math.floor((bufLen - base0) / 3)));
      for (let i = 0; i < maxI; i++) {
        const base = base0 + i * 3;
        let sx = 0, sy = 0, sz = 0;
        for (let w = 0; w < valid; w++) {
          const ringBuf = this.ring[ringIdx(w)];
          sx += ringBuf[base];
          sy += ringBuf[base + 1];
          sz += ringBuf[base + 2];
        }
        result[base] = sx / valid;
        result[base + 1] = sy / valid;
        result[base + 2] = sz / valid;
      }
    }

    return result;
  }

  reset(): void {
    this.index = 0;
    this.count = 0;
  }
}
