/**
 * PHASE 7 — delta-based root translation.
 *
 * ## Why this is a DELTA and not an absolute position
 *
 * `normalizeLandmarks` subtracts the shoulder midpoint from every landmark and
 * divides by the shoulder-to-shoulder distance. Two consequences, both measured
 * on the real fixtures:
 *
 *   1. The shoulder midpoint is EXACTLY [0,0,0] in every normalized buffer, by
 *      construction. Absolute position in the room is destroyed before the
 *      retargeter ever sees it, so no amount of scaling can recover "the signer
 *      walked two metres to the left". Recovering that would require the RAW,
 *      un-normalized image-space landmarks.
 *   2. What survives is the hips' offset RELATIVE to the shoulder anchor. Within
 *      one continuous capture that residual is dominated by monocular depth
 *      noise, which is roughly constant frame-to-frame, so the frame-to-frame
 *      CHANGE is usable even though the absolute value is not.
 *
 * So the root offset is `(hips_now - hips_baseline) * metresPerShoulderWidth`:
 * a change from a baseline, in metres. Applying `hips_now` directly (the old
 * behaviour) is what threw the avatar ~1.6 units below the frame and made it
 * drift, which is why the root translation lock existed.
 *
 * ## Units
 *
 * The source unit is one shoulder width. `metresPerShoulderWidth` converts it
 * to Three.js metres using the avatar's own measured shoulder span, so the
 * mapping is derived from the model rather than guessed.
 *
 * ## Floor
 *
 * Xbot's bind pose already has its toes on the floor (lowest bone y ~= 0), so
 * any DOWNWARD root motion would push the feet through it. Vertical travel is
 * therefore asymmetric: a small lift/crouch band upward, and downward travel
 * clamped so the bind-floor contact is preserved. This is a translation bound,
 * not an IK constraint — no joint angles are altered to reach the floor.
 */
import { Vector3 } from "three";

export interface RootTranslationConfig {
  /** Metres per source shoulder-width unit. Measured from the avatar. */
  metresPerShoulderWidth: number;
  /** Max |x| root travel, metres. Lateral sway is real signal. */
  maxLateralX: number;
  /** Max |z| root travel, metres. Depth is monocular-noise dominated. */
  maxLateralZ: number;
  /** Max upward root travel, metres. */
  maxLift: number;
  /** Max downward root travel, metres. */
  maxDrop: number;
  /** Per-frame slew limit, metres/frame, to reject single-frame spikes. */
  maxStep: number;
}

/**
 * Limits are PER-AXIS and asymmetric on purpose, from measurements on four real
 * 30 fps sign clips (delta of the hip midpoint from the sequence's first frame,
 * in shoulder-width units):
 *
 *   label        dx                  dy                  dz
 *   YES          -0.073 .. +0.010    -0.099 .. +0.142    -0.576 .. +0.587
 *   HELLO        -0.038 .. +0.109     0.000 .. +0.569    -0.499 .. +1.610
 *   THANK_YOU    -0.000 .. +0.162    -0.243 .. +0.030    -1.194 .. +0.369
 *   MOTHER       -0.130 .. +0.034    -1.176 .. +0.031    -0.902 .. +0.145
 *
 * dx is small and physically plausible (a seated signer's weight shift), so it
 * passes through nearly unfiltered. dz swings by up to 1.61 shoulder-widths
 * (~0.49 m) for a SEATED signer, which cannot be real locomotion -- it is
 * monocular depth noise, so Z is deliberately over-constrained. dy has large
 * negative excursions that would push the feet through the floor, so downward
 * travel is clamped to zero.
 */
export const DEFAULT_ROOT_TRANSLATION: RootTranslationConfig = {
  // Xbot's measured LeftArm<->RightArm shoulder span is 0.3033 m, and that span
  // is exactly the quantity normalizeLandmarks divides by.
  metresPerShoulderWidth: 0.3033,
  // Measured lateral max is 0.162 shoulder-widths = 0.049 m; 0.20 m leaves
  // headroom without permitting a visible slide.
  maxLateralX: 0.2,
  // Depth is noise: keep the avatar from sliding toward/away from the camera.
  maxLateralZ: 0.1,
  maxLift: 0.1,
  // Xbot's lowest bind bone (LeftToe_End) sits at y = -0.0029 m, so any
  // downward root motion would push the feet through the floor. Zero drop.
  maxDrop: 0.0,
  // 0.05 m/frame at 30 fps = 1.5 m/s: enough for real motion while rejecting
  // single-frame depth spikes.
  maxStep: 0.05,
};

export interface RootTranslationDebug {
  /** Raw source-space hips position, shoulder-width units. */
  raw: Vector3;
  /** Baseline captured from the first sample of the sequence. */
  baseline: Vector3;
  /** Unclamped delta in metres. */
  uncapped: Vector3;
  /** Final clamped offset in metres (what the model position becomes). */
  offset: Vector3;
  /** True on the first call for a sequence. */
  captured: boolean;
}

export class RootTranslationSolver {
  private readonly cfg: RootTranslationConfig;
  private readonly baseline = new Vector3();
  private readonly lastOffset = new Vector3();
  private hasBaseline = false;
  /**
   * PHASE 10 — per-frame scratch and a reused debug record.
   *
   * `solve()` runs once per rendered frame, and it used to allocate 6 Vector3s
   * per call (the delta, a clone for the slew step, and four for the returned
   * debug record). These are instance fields now; the arithmetic is unchanged.
   *
   * NOTE: `solve()` returns the SAME `debug` object every call, so a caller that
   * needs to retain it across frames must copy the fields. The verify harness
   * reads the numbers immediately, which is unaffected.
   */
  private readonly scratchUncapped = new Vector3();
  private readonly scratchStep = new Vector3();
  private readonly debug: RootTranslationDebug = {
    raw: new Vector3(), baseline: new Vector3(),
    uncapped: new Vector3(), offset: new Vector3(), captured: false,
  };

  constructor(cfg: Partial<RootTranslationConfig> = {}) {
    this.cfg = { ...DEFAULT_ROOT_TRANSLATION, ...cfg };
  }

  /** Start a new sequence: the next sample becomes the baseline. */
  reset(): void {
    this.hasBaseline = false;
    this.lastOffset.set(0, 0, 0);
  }

  get baselineCaptured(): boolean {
    return this.hasBaseline;
  }

  /**
   * Feed one frame's raw (source-unit) hips position.
   * Writes the metre-space root offset into `out` and returns debug detail.
   */
  solve(raw: Vector3, out: Vector3): RootTranslationDebug {
    if (!this.hasBaseline) {
      this.baseline.copy(raw);
      this.hasBaseline = true;
      this.lastOffset.set(0, 0, 0);
      out.set(0, 0, 0);
      this.debug.raw.copy(raw);
      this.debug.baseline.copy(this.baseline);
      this.debug.uncapped.set(0, 0, 0);
      this.debug.offset.set(0, 0, 0);
      this.debug.captured = true;
      return this.debug;
    }

    // ---- the delta, in source units, then converted to metres -----------
    // PHASE 10: `uncapped` and `step` are instance scratch rather than
    // per-call allocations. The arithmetic is byte-for-byte unchanged.
    const uncapped = this.scratchUncapped.set(
      (raw.x - this.baseline.x) * this.cfg.metresPerShoulderWidth,
      (raw.y - this.baseline.y) * this.cfg.metresPerShoulderWidth,
      (raw.z - this.baseline.z) * this.cfg.metresPerShoulderWidth
    );

    // ---- clamp to a human movement envelope (per-axis) --------------------
    uncapped.x = Math.max(-this.cfg.maxLateralX, Math.min(this.cfg.maxLateralX, uncapped.x));
    uncapped.z = Math.max(-this.cfg.maxLateralZ, Math.min(this.cfg.maxLateralZ, uncapped.z));
    uncapped.y = Math.max(-this.cfg.maxDrop, Math.min(this.cfg.maxLift, uncapped.y));

    // ---- slew limit: a single frame cannot teleport the avatar ----------
    const step = this.scratchStep.copy(uncapped).sub(this.lastOffset);
    const stepLen = step.length();
    if (stepLen > this.cfg.maxStep) {
      step.multiplyScalar(this.cfg.maxStep / stepLen);
      uncapped.copy(this.lastOffset).add(step);
    }

    // ---- never let the root go non-finite --------------------------------
    if (!Number.isFinite(uncapped.x) || !Number.isFinite(uncapped.y) || !Number.isFinite(uncapped.z)) {
      uncapped.set(0, 0, 0);
    }

    this.lastOffset.copy(uncapped);
    out.copy(uncapped);

    this.debug.raw.copy(raw);
    this.debug.baseline.copy(this.baseline);
    this.debug.uncapped.copy(uncapped);
    this.debug.offset.copy(uncapped);
    this.debug.captured = false;
    return this.debug;
  }
}
