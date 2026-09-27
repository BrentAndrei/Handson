import type { NormalizedLandmark } from "./types";

// This module smooths landmarks for DISPLAY ONLY. The raw per-frame buffer
// that feeds packHolisticResult -> normalizeLandmarks (and from there any
// downstream model/recording consumer) is deliberately left untouched —
// smoothing or holding stale points there would inject lag and fabricated
// positions into a feature pipeline that's supposed to reflect exactly what
// the model detected each frame. Two independent problems, two independent
// fixes:
//
//   1. Jitter: raw per-frame detector noise was drawn straight to canvas
//      with no temporal filtering. Fixed with a One Euro Filter per point.
//   2. Landmarks vanishing: fast motion causes the detector to occasionally
//      miss a hand/pose for a frame or two (motion blur). The old draw path
//      had no memory, so a single missed frame made the overlay blink out
//      and pop back in. Fixed by holding the last smoothed pose briefly and
//      fading it out, instead of hard-cutting to nothing.

/**
 * One Euro Filter (Casiez, Roussel, Vogel — 2012).
 * Low-pass filters a noisy scalar signal while adapting its cutoff to the
 * signal's own speed: near-stationary points get filtered hard (kills
 * jitter), fast-moving points get filtered lightly (kills lag). This is
 * the standard technique for smoothing tracked landmarks/cursors and is
 * what MediaPipe's own reference demos use for the same purpose.
 */
class LowPassFilter {
  private y: number | null = null;
  private s = 0;

  filter(value: number, alpha: number): number {
    if (this.y === null) {
      this.s = value;
    } else {
      this.s = alpha * value + (1 - alpha) * this.s;
    }
    this.y = value;
    return this.s;
  }

  lastRaw(): number | null {
    return this.y;
  }

  reset(): void {
    this.y = null;
    this.s = 0;
  }
}

function smoothingAlpha(cutoff: number, dtSeconds: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dtSeconds);
}

class OneEuroFilter {
  private readonly minCutoff: number;
  private readonly beta: number;
  private readonly dCutoff: number;
  private readonly xFilter = new LowPassFilter();
  private readonly dxFilter = new LowPassFilter();
  private lastTimeMs: number | null = null;

  constructor(minCutoff: number, beta: number, dCutoff: number) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  filter(value: number, timestampMs: number): number {
    if (this.lastTimeMs === null) {
      this.lastTimeMs = timestampMs;
      this.xFilter.filter(value, 1);
      this.dxFilter.filter(0, 1);
      return value;
    }

    // Guard against a zero/negative delta (duplicate timestamp, clock
    // weirdness) which would otherwise divide by ~0 inside the cutoff calc.
    let dt = (timestampMs - this.lastTimeMs) / 1000;
    if (!(dt > 0)) dt = 1 / 30;
    this.lastTimeMs = timestampMs;

    const prevRaw = this.xFilter.lastRaw() ?? value;
    const derivative = (value - prevRaw) / dt;
    const smoothedDerivative = this.dxFilter.filter(
      derivative,
      smoothingAlpha(this.dCutoff, dt)
    );

    const adaptiveCutoff = this.minCutoff + this.beta * Math.abs(smoothedDerivative);
    return this.xFilter.filter(value, smoothingAlpha(adaptiveCutoff, dt));
  }

  reset(): void {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTimeMs = null;
  }
}

export interface SmoothingOptions {
  /** Lower = smoother when nearly still, but more lag on slow drift. */
  minCutoff: number;
  /** Higher = snappier tracking of fast motion, at the cost of some jitter during it. */
  beta: number;
  dCutoff: number;
  /** How long (ms) to keep showing the last known pose after detection drops, fading out. */
  holdMs: number;
  /** How long (ms) of continuous absence before filters reset (avoids a stale-velocity snap on return). */
  resetMs: number;
  /** How long (ms) to ramp opacity 0->1 when a group starts being detected again, mirroring the fade-out on loss instead of snapping straight to fully visible. */
  fadeInMs: number;
}

const DEFAULT_OPTIONS: SmoothingOptions = {
  minCutoff: 1.0,
  beta: 0.4,
  dCutoff: 1.0,
  // Was 250ms, tuned up: lowering the detector's confidence thresholds
  // (see useHolisticPipeline.ts) mostly fixed outright misses, but a
  // held-still hand can still flicker for a couple of consecutive frames
  // near the confidence boundary rather than just one. 400ms covers a
  // multi-frame gap at 24fps (~10 frames) without masking a hand that's
  // actually left the frame for good.
  holdMs: 400,
  resetMs: 900,
  // Deliberately shorter than holdMs — arriving should read as prompt, not
  // sluggish, while still softening the "snaps into existence" pop that a
  // hard cut to opacity 1 produces, especially right as a bridge connector
  // (see drawResult.ts) is also fading into existence alongside it.
  fadeInMs: 120,
};

export interface SmoothedGroup {
  landmarks: NormalizedLandmark[] | null;
  /** 1 = fully live detection, (0,1) = held-over from a recent dropout and fading, 0 = not shown. */
  opacity: number;
}

/** Smooths one landmark group (pose, a hand, or face) across frames. */
class GroupSmoother {
  private readonly opts: SmoothingOptions;
  private filtersX: OneEuroFilter[] = [];
  private filtersY: OneEuroFilter[] = [];
  private filtersZ: OneEuroFilter[] = [];
  private held: NormalizedLandmark[] | null = null;
  private lastSeenMs = -Infinity;
  // When the current uninterrupted visible streak started — null while
  // nothing is being shown. Drives the fade-in ramp below; distinct from
  // lastSeenMs, which tracks the most recent detection regardless of
  // whether display was continuous.
  private visibleSinceMs: number | null = null;

  constructor(opts: SmoothingOptions) {
    this.opts = opts;
  }

  private ensureCapacity(count: number): void {
    while (this.filtersX.length < count) {
      this.filtersX.push(new OneEuroFilter(this.opts.minCutoff, this.opts.beta, this.opts.dCutoff));
      this.filtersY.push(new OneEuroFilter(this.opts.minCutoff, this.opts.beta, this.opts.dCutoff));
      this.filtersZ.push(new OneEuroFilter(this.opts.minCutoff, this.opts.beta, this.opts.dCutoff));
    }
  }

  update(landmarks: NormalizedLandmark[] | undefined, timestampMs: number): SmoothedGroup {
    if (landmarks && landmarks.length > 0) {
      const wasShowing = this.held !== null;
      if (timestampMs - this.lastSeenMs > this.opts.resetMs) {
        for (const f of this.filtersX) f.reset();
        for (const f of this.filtersY) f.reset();
        for (const f of this.filtersZ) f.reset();
      }

      this.ensureCapacity(landmarks.length);

      const smoothed: NormalizedLandmark[] = new Array(landmarks.length);
      for (let i = 0; i < landmarks.length; i++) {
        const p = landmarks[i];
        smoothed[i] = {
          x: this.filtersX[i].filter(p.x, timestampMs),
          y: this.filtersY[i].filter(p.y, timestampMs),
          z: this.filtersZ[i].filter(p.z, timestampMs),
          visibility: p.visibility,
        } as NormalizedLandmark;
      }

      if (!wasShowing) {
        this.visibleSinceMs = timestampMs;
      }
      this.held = smoothed;
      this.lastSeenMs = timestampMs;

      const streakAge = timestampMs - (this.visibleSinceMs ?? timestampMs);
      const opacity =
        this.opts.fadeInMs > 0 ? Math.min(1, streakAge / this.opts.fadeInMs) : 1;
      return { landmarks: smoothed, opacity };
    }

    if (this.held && timestampMs - this.lastSeenMs <= this.opts.holdMs) {
      const age = timestampMs - this.lastSeenMs;
      const opacity = Math.max(0, 1 - age / this.opts.holdMs);
      return { landmarks: this.held, opacity };
    }

    this.held = null;
    this.visibleSinceMs = null;
    return { landmarks: null, opacity: 0 };
  }

  /**
   * Same as update(), but does NOT advance lastSeenMs/visibleSinceMs.
   * Used for extrapolated/interpolated frames between real detections:
   * the overlay position advances, but the dropout hold/fade timers keep
   * counting from the last real detection so a genuine miss still fades out
   * correctly.
   */
  updateExtrapolated(landmarks: NormalizedLandmark[] | undefined, timestampMs: number): SmoothedGroup {
    if (!landmarks || landmarks.length === 0) {
      // Still run the normal hold/fade check against the real lastSeenMs.
      if (this.held && timestampMs - this.lastSeenMs <= this.opts.holdMs) {
        const age = timestampMs - this.lastSeenMs;
        const opacity = Math.max(0, 1 - age / this.opts.holdMs);
        return { landmarks: this.held, opacity };
      }
      this.held = null;
      this.visibleSinceMs = null;
      return { landmarks: null, opacity: 0 };
    }

    const wasShowing = this.held !== null;
    this.ensureCapacity(landmarks.length);

    const smoothed: NormalizedLandmark[] = new Array(landmarks.length);
    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      smoothed[i] = {
        x: this.filtersX[i].filter(p.x, timestampMs),
        y: this.filtersY[i].filter(p.y, timestampMs),
        z: this.filtersZ[i].filter(p.z, timestampMs),
        visibility: p.visibility,
      } as NormalizedLandmark;
    }

    if (!wasShowing) {
      this.visibleSinceMs = timestampMs;
    }
    this.held = smoothed;
    // Intentionally do NOT update this.lastSeenMs here.

    const streakAge = timestampMs - (this.visibleSinceMs ?? timestampMs);
    const opacity =
      this.opts.fadeInMs > 0 ? Math.min(1, streakAge / this.opts.fadeInMs) : 1;
    return { landmarks: smoothed, opacity };
  }
}

export interface SmoothedHolisticResult {
  pose: SmoothedGroup;
  leftHand: SmoothedGroup;
  rightHand: SmoothedGroup;
  face: SmoothedGroup;
}

/** Minimal shape this module needs from a HolisticLandmarkerResult. */
export interface HolisticLandmarksInput {
  poseLandmarks: NormalizedLandmark[][];
  leftHandLandmarks: NormalizedLandmark[][];
  rightHandLandmarks: NormalizedLandmark[][];
  faceLandmarks: NormalizedLandmark[][];
}

/**
 * Owns per-group smoothing state across the whole session. One instance
 * should live for the lifetime of the capture loop (create once, call
 * update() every frame) — recreating it every frame would defeat the
 * filters' whole purpose, since they rely on history.
 */
export class HolisticSmoother {
  private readonly pose: GroupSmoother;
  private readonly leftHand: GroupSmoother;
  private readonly rightHand: GroupSmoother;
  private readonly face: GroupSmoother;

  constructor(overrides: Partial<SmoothingOptions> = {}) {
    const base: SmoothingOptions = { ...DEFAULT_OPTIONS, ...overrides };
    // Hands move faster and more erratically than pose/face, so give them a
    // higher beta: still smooth when idle, but they won't visibly lag
    // behind a fast gesture the way the base settings would.
    const handOpts: SmoothingOptions = { ...base, beta: base.beta * 1.75 };

    this.pose = new GroupSmoother(base);
    this.leftHand = new GroupSmoother(handOpts);
    this.rightHand = new GroupSmoother(handOpts);
    this.face = new GroupSmoother(base);
  }

  update(result: HolisticLandmarksInput, timestampMs: number): SmoothedHolisticResult {
    return {
      pose: this.pose.update(result.poseLandmarks[0], timestampMs),
      leftHand: this.leftHand.update(result.leftHandLandmarks[0], timestampMs),
      rightHand: this.rightHand.update(result.rightHandLandmarks[0], timestampMs),
      face: this.face.update(result.faceLandmarks[0], timestampMs),
    };
  }

  /**
   * Feed extrapolated/interpolated landmarks into the display smoothers
   * without advancing the real-detection dropout timers. Use this for
   * frames fabricated between two actual detections so the overlay keeps
   * moving while the smoother still knows when the last *real* detection
   * was for hold/fade purposes.
   */
  updateExtrapolated(result: HolisticLandmarksInput, timestampMs: number): SmoothedHolisticResult {
    return {
      pose: this.pose.updateExtrapolated(result.poseLandmarks[0], timestampMs),
      leftHand: this.leftHand.updateExtrapolated(result.leftHandLandmarks[0], timestampMs),
      rightHand: this.rightHand.updateExtrapolated(result.rightHandLandmarks[0], timestampMs),
      face: this.face.updateExtrapolated(result.faceLandmarks[0], timestampMs),
    };
  }
}
