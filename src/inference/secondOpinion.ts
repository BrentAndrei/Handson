/**
 * DTW second opinion for gray-zone emissions.
 *
 * The BiGRU emits only when the ring-mean confidence, margin,
 * majority vote and debounce all pass (temporalEnsemble.ts).
 * Between MIN_CONFIDENCE (0.5) and a confident 0.7, though,
 * the model is guessing — and the weakest classes have as few
 * as 4 training samples, so their probabilities are statistics
 * on almost no data.
 *
 * This module consults a different, geometry-based signal for
 * exactly those emissions: the app already ships every sign's
 * canonical recording (public/avatar-data/) and the DTW engine
 * that powers the Practice tutor (signSimilarity.ts). When the
 * ensemble emits from a gray zone, the live frames are scored
 * against the canonical recordings of the ensemble's top-3
 * candidates, and the model probabilities are re-ranked against
 * the DTW scores. A switch only happens when DTW is itself
 * confident and the combined score clearly wins — the second
 * opinion can demote a flicker, never fabricate a label that
 * the model didn't itself rank in its top 3.
 *
 * Pure decision logic (applySecondOpinion / needsSecondOpinion)
 * is separated from the async loading (CanonicalStore /
 * scoreCandidates) so the gate can test it without a browser.
 */
import { SignSimilarityTracker } from "../features/signSimilarity";
import { TOTAL_FLOATS } from "../mediapipe/types";

/** Consult DTW below this ring-mean confidence. Anchored to
 *  the display threshold (PredictionOverlay shows 0.6) — below
 *  0.7 the label is uncertain enough to be worth a second look. */
export const SECOND_OPINION_CONFIDENCE = 0.7;
/** ...or when the top-1/top-2 gap is this thin (the ensemble
 *  itself only requires 0.1). */
export const SECOND_OPINION_MARGIN = 0.15;
/** How many of the ensemble's top candidates DTW scores. */
export const SECOND_OPINION_TOP_N = 3;
/** Absolute DTW floor the winner must clear. Deliberately
 *  low: the consult scores the live window's TAIL against
 *  the full canonical recording, so even the correct class
 *  scores far below the Practice tutor's start-aligned
 *  ~1.0 (measured 0.15 on real corpus data). What matters
 *  for re-ranking is the relative ordering, guarded below. */
export const DTW_FLOOR = 0.1;
/** The DTW winner must beat the model top-1's DTW score by
 *  at least this ratio. Both candidates pay the same
 *  tail-alignment penalty against the same live window, so
 *  the ratio is alignment-invariant — a 30-frame sign and a
 *  3000-frame sign are compared on equal footing. */
export const DTW_RATIO = 1.6;
/** Combined score must beat the original by at least this. */
export const SECOND_OPINION_GAIN = 0.05;
/** Model probability vs DTW score in the combined ranking. */
export const MODEL_WEIGHT = 0.5;
/** Live frames scored per consult (~1s at 30fps — the same
 *  evidence span as the ensemble's 7-window ring). */
export const MAX_LIVE_FRAMES = 32;
/** Soft cap on cached canonical recordings (~a few MB each). */
export const MAX_CACHED_CANONICALS = 64;

export interface CandidateProb {
  label: string;
  confidence: number;
}

export interface SecondOpinionDecision {
  /** Final label after re-ranking. */
  label: string;
  /** Model confidence when unchanged; the combined
   *  model+DTW score (0..1) when switched. */
  confidence: number;
  switched: boolean;
  originalLabel: string;
  /** DTW scores by (raw) label — the evidence, for logging. */
  scores: Record<string, number>;
}

/** Is this emission uncertain enough to consult DTW? */
export function needsSecondOpinion(
  confidence: number,
  margin: number
): boolean {
  return (
    confidence < SECOND_OPINION_CONFIDENCE || margin < SECOND_OPINION_MARGIN
  );
}

/**
 * Re-rank the ensemble's top candidates against DTW scores.
 * Pure and synchronous — the caller resolves the DTW scores
 * first (scoreCandidates). Returns the decision, switched or
 * not; never returns a label outside `top3`.
 */
export function applySecondOpinion(
  top3: CandidateProb[],
  dtwScores: Record<string, number>
): SecondOpinionDecision {
  const combined = (c: CandidateProb): number =>
    MODEL_WEIGHT * c.confidence + (1 - MODEL_WEIGHT) * (dtwScores[c.label] ?? 0);

  let winner = top3[0];
  for (const candidate of top3.slice(1)) {
    if (combined(candidate) > combined(winner)) winner = candidate;
  }
  const base = top3[0];
  if (winner.label === base.label) {
    return {
      label: base.label,
      confidence: base.confidence,
      switched: false,
      originalLabel: base.label,
      scores: dtwScores,
    };
  }
  const winnerDtw = dtwScores[winner.label] ?? 0;
  const baseDtw = dtwScores[base.label] ?? 0;
  // DTW must have found a real alignment for the winner...
  const floorOk = winnerDtw >= DTW_FLOOR;
  // ...and must beat the model's top-1 by a clear ratio.
  // (When the top-1 has no recording at all, the floor
  // alone decides.)
  const ratioOk =
    baseDtw > 0 ? winnerDtw >= DTW_RATIO * baseDtw : floorOk;
  const gain = combined(winner) - combined(base);

  if (floorOk && ratioOk && gain >= SECOND_OPINION_GAIN) {
    return {
      label: winner.label,
      confidence: combined(winner),
      switched: true,
      originalLabel: base.label,
      scores: dtwScores,
    };
  }
  return {
    label: base.label,
    confidence: base.confidence,
    switched: false,
    originalLabel: base.label,
    scores: dtwScores,
  };
}

/**
 * Fetch one sign's canonical recording as raw packed 345-float
 * frames. Unlike signSimilarity's loadCanonicalRawFrames (which
 * only tries the underscore-sanitized filename), this tries the
 * URL-encoded label first — corpus files keep their punctuation
 * ("ALRIGHT, NO PROBLEM.json", "I'M OKAY.json") — then falls
 * back to the sanitized form. Returns null when there is no
 * recording for the label.
 */
async function fetchCanonicalFrames(
  label: string
): Promise<Float32Array[] | null> {
  const urls = [
    `/avatar-data/${encodeURIComponent(label)}.json`,
    `/avatar-data/${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`,
  ];
  for (const url of urls) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) continue;
      const data = (await resp.json()) as { frames?: unknown };
      if (!data || !Array.isArray(data.frames)) continue;
      const flat = new Float32Array(data.frames as unknown as number[]);
      const frameCount = Math.floor(flat.length / TOTAL_FLOATS);
      if (frameCount === 0) continue;
      const frames: Float32Array[] = [];
      for (let i = 0; i < frameCount; i++) {
        frames.push(flat.slice(i * TOTAL_FLOATS, (i + 1) * TOTAL_FLOATS));
      }
      return frames;
    } catch {
      // Next URL form, then null.
    }
  }
  return null;
}

/** Lazy, memoized cache of canonical recordings. */
export class CanonicalStore {
  private cache = new Map<string, Promise<Float32Array[] | null>>();

  load(label: string): Promise<Float32Array[] | null> {
    const cached = this.cache.get(label);
    if (cached) return cached;
    if (this.cache.size >= MAX_CACHED_CANONICALS) this.cache.clear();
    const pending = fetchCanonicalFrames(label).then(
      (frames) => frames,
      () => {
        // Don't cache the failure — a later consult can retry.
        this.cache.delete(label);
        return null;
      }
    );
    this.cache.set(label, pending);
    return pending;
  }
}

/** Default store shared by the inference hook. */
export const canonicalStore = new CanonicalStore();

/**
 * Score the live frames against each candidate's canonical
 * recording with the DTW engine. Returns DTW scores (0..1) for
 * the candidates that have recordings; candidates without one
 * are simply absent from the result.
 */
export async function scoreCandidates(
  labels: string[],
  liveFrames: Float32Array[],
  store: { load(label: string): Promise<Float32Array[] | null> } = canonicalStore
): Promise<Record<string, number>> {
  const scores: Record<string, number> = {};
  await Promise.all(
    labels.map(async (label) => {
      const canonical = await store.load(label);
      if (!canonical || canonical.length === 0) return;
      const tracker = new SignSimilarityTracker(label, canonical);
      const count = Math.min(liveFrames.length, MAX_LIVE_FRAMES);
      for (let i = liveFrames.length - count; i < liveFrames.length; i++) {
        tracker.pushLiveFrame(liveFrames[i]);
      }
      const result = tracker.finish();
      if (result) scores[label] = result.score;
    })
  );
  return scores;
}
