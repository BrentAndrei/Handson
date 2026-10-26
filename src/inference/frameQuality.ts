/**
 * Frame quality gate for streaming sign inference.
 *
 * The BiGRU classifies fixed 16-frame windows, so a single
 * garbage frame inside a window corrupts the whole prediction.
 * MediaPipe produces those frames routinely: a hand leaves the
 * frame, the subject turns, or a landmark jumps. Nothing in the
 * current pipeline checks for this — every captured frame is
 * pushed into the sliding window regardless of quality.
 *
 * This module evaluates each frame BEFORE it enters the
 * inference buffer and exposes three signals:
 *
 * 1. `usable` — should this frame be pushed at all?
 *    A frame is unusable when BOTH hand groups are absent or
 *    collapsed (spread below the presence threshold), because
 *    the model would then see a window that transitions from a
 *    real sign into nothing.
 *
 * 2. `isRest` — is the user signing at all?
 *    Consecutive hand-absent frames mean the signer has
 *    stopped. The caller should reset the window buffer and the
 *    temporal ensemble so a later sign is not contaminated by
 *    stale frames, and so a held rest pose cannot fire a
 *    prediction.
 *
 * 3. `motion` — how much did the hands move this frame?
 *    Used for jitter detection (a huge single-frame jump is a
 *    tracking dropout, not a sign) and available for adaptive
 *    stride.
 *
 * Spread is measured in the same shoulder-width units
 * normalizeLandmarks produces, so the thresholds are invariant
 * to how far the signer sits from the camera.
 *
 * Pure and synchronous — no timers, no DOM. The clock is
 * injectable so tests can drive rest-state transitions
 * deterministically.
 */

export interface FrameQuality {
  /** True when the frame is good enough to feed the model. */
  usable: boolean;
  /** True when no usable hand is present — the signer is at rest. */
  isRest: boolean;
  /** Hand motion this frame, in shoulder-width units (0..n). */
  motion: number;
  /** Human-readable reasons the frame was rejected. */
  issues: string[];
}

/** A hand group whose 21 landmarks span less than this (in
 *  shoulder-width units) is either absent or collapsed. Mirrors
 *  MIN_HAND_SPREAD in mediapipeAdapter.ts — an absent group
 *  normalizes to 21 IDENTICAL points (spread 0), so this is the
 *  shift-invariant presence test. */
export const MIN_HAND_SPREAD = 0.02;

/** A single-frame hand jump larger than this (shoulder widths)
 *  is a tracking dropout, not part of a sign. FSL wrist motion
 *  is fast but continuous; a >0.5 jump in one frame is a
 *  glitch. */
export const MAX_FRAME_JUMP = 0.5;

/** Consecutive rest frames before the gate reports isRest.
 *  One absent frame is a blink; several in a row means the signer
 *  actually stopped. Raised to 6 because inference is fed at ~8 Hz
 *  (time-matched to training) so a mid-sign hand flicker must NOT
 *  wipe the whole sliding window. */
export const REST_FRAME_THRESHOLD = 6;

/** Offsets of the two hand groups in a selectFeatures() output. */
const LEFT_HAND_START = 99;
const RIGHT_HAND_START = 162;
const HAND_COUNT = 21;

/** Max distance from a hand group's centroid to any landmark. */
export function handGroupSpread(
  data: ArrayLike<number>,
  start: number
): number {
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < HAND_COUNT; i++) {
    const b = start + i * 3;
    cx += data[b];
    cy += data[b + 1];
    cz += data[b + 2];
  }
  cx /= HAND_COUNT;
  cy /= HAND_COUNT;
  cz /= HAND_COUNT;

  let maxSq = 0;
  for (let i = 0; i < HAND_COUNT; i++) {
    const b = start + i * 3;
    const dx = data[b] - cx;
    const dy = data[b + 1] - cy;
    const dz = data[b + 2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > maxSq) maxSq = d2;
  }
  return Math.sqrt(maxSq);
}

/** Centroid-to-centroid hand displacement between two frames. */
function handMotion(
  prev: ArrayLike<number>,
  cur: ArrayLike<number>
): number {
  const centroid = (data: ArrayLike<number>, start: number): [number, number, number] => {
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < HAND_COUNT; i++) {
      const b = start + i * 3;
      x += data[b]; y += data[b + 1]; z += data[b + 2];
    }
    return [x / HAND_COUNT, y / HAND_COUNT, z / HAND_COUNT];
  };
  let sum = 0;
  for (const start of [LEFT_HAND_START, RIGHT_HAND_START]) {
    const a = centroid(prev, start);
    const b = centroid(cur, start);
    sum += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }
  return sum;
}

export class FrameQualityGate {
  private prevFrame: Float32Array | null = null;
  private restFrames = 0;

  /**
   * Evaluate one feature frame (selectFeatures output).
   * Call before pushing into the inference buffer.
   */
  evaluate(frame: Float32Array): FrameQuality {
    const issues: string[] = [];

    const leftSpread = handGroupSpread(frame, LEFT_HAND_START);
    const rightSpread = handGroupSpread(frame, RIGHT_HAND_START);
    const leftPresent = leftSpread >= MIN_HAND_SPREAD;
    const rightPresent = rightSpread >= MIN_HAND_SPREAD;
    const anyPresent = leftPresent || rightPresent;

    if (!anyPresent) {
      issues.push("no-hands");
    }

    // Motion vs the previous frame.
    let motion = 0;
    if (this.prevFrame && this.prevFrame.length === frame.length) {
      motion = handMotion(this.prevFrame, frame);
      if (motion > MAX_FRAME_JUMP) {
        issues.push(`jump ${motion.toFixed(2)}`);
      }
    }

    // Rest-state hysteresis: one absent frame is a blink, several
    // in a row is the signer stopping.
    if (anyPresent) {
      this.restFrames = 0;
    } else {
      this.restFrames++;
    }
    const isRest = this.restFrames >= REST_FRAME_THRESHOLD;

    // A frame is usable when at least one hand is present. A
    // single jitter frame is still pushed — the window buffer
    // and ensemble smooth isolated spikes — but it is flagged so
    // the caller can decide.
    const usable = anyPresent;

    this.prevFrame = frame;
    return { usable, isRest, motion, issues };
  }

  /** Forget history (camera restarted, view changed). */
  reset(): void {
    this.prevFrame = null;
    this.restFrames = 0;
  }
}
