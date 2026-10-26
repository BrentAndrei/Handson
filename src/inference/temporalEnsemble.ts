/**
 * Temporal ensemble for streaming sign inference.
 *
 * The BiGRU classifies overlapping 16-frame windows every 6
 * frames, so a single ~1s sign produces many probability
 * vectors. Deciding on 2 consecutive argmaxes (the previous
 * scheme) discards almost all of that evidence: a 0.51-vs-0.49
 * window counts the same as a 0.95-vs-0.01 one, and two
 * blips in a row fire a prediction.
 *
 * This module averages the probability vectors over a ring of
 * the last ENSEMBLE_WINDOWS windows (temporal softmax
 * averaging — the standard fix for windowed classifiers) and
 * emits a label only when ALL of the following hold:
 *   1. The ring is full (enough evidence accumulated).
 *   2. Mean top-class probability >= MIN_CONFIDENCE.
 *   3. Mean top-class beats the runner-up by CONFIDENCE_MARGIN.
 *   4. The top class is the per-window argmax in at least
 *      MIN_VOTE_FRACTION of the ring (majority vote — a single
 *      noisy window can't swing the result).
 *   5. The label wasn't just emitted (EMISSION_HOLD_MS debounce,
 *      unless the label changed — rapid successive signs still work).
 *
 * After an emission the ring is cleared: a sign is a discrete
 * event, so the next emission must be earned from fresh windows.
 * This also prevents one held sign from re-firing continuously.
 *
 * Pure and synchronous (the clock is injectable for tests) —
 * the React hook owns the async tf.js readback and feeds each
 * window's full probability vector into push().
 */

/** Per-window confidence floor, applied to the RING MEAN.
 *  Adjusted from 0.10 to 0.03 based on measured canonical-data
 *  softmax values (mean ring confidence ~0.09–0.13). Keeping it
 *  above pure-noise floor (~1/399 ≈ 0.0025) so that sporadic
 *  false-emissions are limited to unambiguous winner cases. */
export const MIN_CONFIDENCE = 0.03;
/** Minimum mean-probability gap between top-1 and top-2.
 *  Adjusted from 0.05 to 0.005: the model's probability distribution
 *  is relatively flat across 399 classes, so a very narrow margin
 *  prevents legitimate signs from being rejected. */
export const CONFIDENCE_MARGIN = 0.005;
/** How many window results vote. At DEFAULT_STRIDE=3 frames
 *  gated at 8 Hz: 5 windows ≈ 1.9s of evidence — sufficient for
 *  a ~1s sign, while 7 windows (~2.6s) were too long a hold.
 *  The vote fraction check is still enforced on the ring. */
export const ENSEMBLE_WINDOWS = 5;
/** Fraction of the ring that must agree on the top class.
 *  Adjusted from 0.6 to 0.51: a simple majority prevents a single
 *  noisy window from swinging the result while still allowing
 *  genuine signs to emit. */
export const MIN_VOTE_FRACTION = 0.51;
/** Minimum time between emissions of the SAME label.
 *  At 8 Hz inference with stride 3 and 5 windows, a fresh
 *  emission can occur every ~1.5s minimum. Changed debounce
 *  from 1s to still prevent double-firing on held signs. */
export const EMISSION_HOLD_MS = 1000;

/**
 * Labels whose corpus recording is bit-identical to
 * another class (verified 2026-10-11: "ALRIGHT, NO
 * PROBLEM.json" and "NO.json" are the same 38,640
 * floats). The model was trained with both labels on
 * identical data, so it can never separate them —
 * emissions of the duplicate are aliased to the
 * surviving label. Runner-up and top-3 reporting keep
 * the raw labels for confusion analysis, and the
 * second opinion resolves canonical recordings by the
 * raw label (the file on disk keeps its punctuation).
 */
export const EMISSION_ALIASES: Readonly<Record<string, string>> = {
  "ALRIGHT, NO PROBLEM": "NO",
};

/** A label the ensemble decided on, with its mean confidence. */
export interface EnsembleResult {
  label: string;
  confidence: number;
  /** Runner-up in the ring mean — exposed for confusion
   *  analysis in the accuracy logger, not for emission. */
  runnerUp?: { label: string; confidence: number };
  /** Top classes by ring-mean probability, descending
   *  (raw, un-aliased labels — the DTW second opinion
   *  fetches canonical recordings by these names). */
  top3?: { label: string; confidence: number }[];
  /** Ring-mean gap between top-1 and top-2. */
  margin?: number;
}

export class TemporalEnsemble {
  private ring: Float32Array[] = [];
  private lastEmission: { label: string; at: number } | null = null;

  constructor(
    private readonly numClasses: number,
    private readonly labels: string[],
    private readonly maxWindows: number = ENSEMBLE_WINDOWS,
    private readonly minVoteFraction: number = MIN_VOTE_FRACTION,
    private readonly minConfidence: number = MIN_CONFIDENCE,
    private readonly minMargin: number = CONFIDENCE_MARGIN,
    private readonly holdMs: number = EMISSION_HOLD_MS,
    private readonly now: () => number = () => performance.now(),
  ) {}

  /** Number of classes the ensemble was built for (labels.length). */
  get classCount(): number {
    return this.numClasses;
  }

  /** Current number of window probs in the ring (0..ENSEMBLE_WINDOWS). */
  get size(): number {
    return this.ring.length;
  }

  /**
   * Feed one window's full probability vector. Returns a decision
   * when the ensemble rules are all satisfied, otherwise null.
   */
  push(windowProbs: Float32Array): EnsembleResult | null {
    if (windowProbs.length !== this.numClasses) {
      throw new Error(
        `TemporalEnsemble: expected ${this.numClasses} probs, got ${windowProbs.length}`
      );
    }
    this.ring.push(windowProbs);
    if (this.ring.length > this.maxWindows) this.ring.shift();
    if (this.ring.length < this.maxWindows) return null;

    // Mean probability per class across the ring, tracking the
    // top-3 classes in the same pass.
    const mean = new Float32Array(this.numClasses);
    let top1 = 0;
    let top2 = -1;
    let top3Idx = -1;
    for (let c = 0; c < this.numClasses; c++) {
      let sum = 0;
      for (const window of this.ring) sum += window[c];
      mean[c] = sum / this.ring.length;
      if (c === 0) continue;
      if (mean[c] > mean[top1]) {
        top3Idx = top2;
        top2 = top1;
        top1 = c;
      } else if (top2 < 0 || mean[c] > mean[top2]) {
        top3Idx = top2;
        top2 = c;
      } else if (top3Idx < 0 || mean[c] > mean[top3Idx]) {
        top3Idx = c;
      }
    }
    if (top2 < 0) top2 = top1 === 0 ? 1 : 0;
    if (top3Idx < 0 || top3Idx === top1 || top3Idx === top2) {
      top3Idx =
        [0, 1, 2].find(
          (c) => c !== top1 && c !== top2 && c < this.numClasses
        ) ?? -1;
    }

    // Majority vote: how many windows' argmax is the ensemble's top class.
    let votes = 0;
    for (const window of this.ring) {
      let argmax = 0;
      for (let c = 1; c < this.numClasses; c++) {
        if (window[c] > window[argmax]) argmax = c;
      }
      if (argmax === top1) votes++;
    }
    const voteFraction = votes / this.ring.length;

    const rawLabel = this.labels[top1] ?? `class_${top1}`;
    const label = EMISSION_ALIASES[rawLabel] ?? rawLabel;
    const confidence = mean[top1];
    const margin = mean[top1] - mean[top2];
    const debounced =
      this.lastEmission !== null &&
      this.lastEmission.label === label &&
      this.now() - this.lastEmission.at < this.holdMs;

    if (
      confidence >= this.minConfidence &&
      margin >= this.minMargin &&
      voteFraction >= this.minVoteFraction &&
      !debounced
    ) {
      this.lastEmission = { label, at: this.now() };
      this.ring = [];
      // Top-3 keeps RAW (un-aliased) labels: the second
      // opinion fetches canonical recordings by filename.
      const top3: { label: string; confidence: number }[] = [];
      const seen = new Set<number>();
      for (const c of [top1, top2, top3Idx]) {
        if (c < 0 || seen.has(c)) continue;
        seen.add(c);
        top3.push({
          label: this.labels[c] ?? `class_${c}`,
          confidence: mean[c],
        });
      }
      return {
        label,
        confidence,
        margin,
        runnerUp: {
          label: this.labels[top2] ?? `class_${top2}`,
          confidence: mean[top2],
        },
        top3,
      };
    }
    return null;
  }

  /** Drop accumulated evidence (e.g. camera lost / view changed). */
  reset(): void {
    this.ring = [];
  }
}
