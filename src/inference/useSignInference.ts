import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { FEATURE_DIM } from "../features/featureSelect";
import { DEFAULT_STRIDE, SlidingWindowBuffer, WINDOW_SIZE } from "../features/windowBuffer";
import { FrameQualityGate } from "./frameQuality";
import { TemporalEnsemble, EMISSION_ALIASES } from "./temporalEnsemble";
import { getAccuracyLogger } from "./accuracyLog";
import {
  MAX_LIVE_FRAMES,
  applySecondOpinion,
  canonicalStore,
  needsSecondOpinion,
  scoreCandidates,
} from "./secondOpinion";

export interface Prediction {
  label: string;
  confidence: number;
}

// Confidence-gate constants now live in the temporal ensemble
// (temporalEnsemble.ts), which applies them to the ring MEAN
// instead of a single window. Re-exported for compatibility.
export { CONFIDENCE_MARGIN, MIN_CONFIDENCE } from "./temporalEnsemble";

/** Per-emission top-3 logging, off by default (VITE_DEBUG_INFERENCE=1). */
const DEBUG_INFERENCE = import.meta.env.VITE_DEBUG_INFERENCE === "1";

/**
 * Hard ceiling on the DTW second-opinion consult.
 *
 * The consult is a network fetch (the canonical recordings) followed by a
 * DTW alignment. Both can stall: a 404 that never resolves, a slow CDN, or
 * a pathological O(n*m) alignment against a long recording. Before this
 * timeout existed, a stalled consult meant the emission was simply LOST —
 * the `scoreCandidates(...)` promise had no rejection handler either, so a
 * throw inside it produced an unhandled rejection and the user saw NO
 * prediction at all for that sign. That is the "hard time detecting"
 * symptom for every gray-zone emission.
 *
 * With the timeout, the worst case is that the second opinion is simply
 * skipped and the ensemble's own label is emitted.
 */
const SECOND_OPINION_TIMEOUT_MS = 1500;

/** Resolve to `null` if `promise` has not settled within `ms`. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  return new Promise<T | null>((resolve) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      resolve(null);
    }, ms);
    promise.then(
      (value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(value);
      },
      () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(null);
      }
    );
  });
}

/**
 * Loads a tfjs LayersModel trained on [WINDOW_SIZE, FEATURE_DIM] windows
 * (see the Python training script) and runs live inference against a
 * streaming sliding-window buffer. Call pushFrame() from the same
 * per-frame callback that feeds the dataset recorder — it silently no-ops
 * until the model has loaded and the window has filled at least once.
 *
 * `labels` must be in the exact order the model's output layer was
 * trained with (see label_map.json alongside the exported model).
 */
export function useSignInference(modelUrl: string, labels: string[], stride = DEFAULT_STRIDE) {
  const modelRef = useRef<tf.LayersModel | null>(null);
  const bufferRef = useRef(new SlidingWindowBuffer(FEATURE_DIM, WINDOW_SIZE, stride));
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const pushCountRef = useRef(0);
  const restCountRef = useRef(0);
  const unusableCountRef = useRef(0);
  const emitCountRef = useRef(0);
  const lastEmitTimeRef = useRef(0);
  // Temporal ensemble: averages the probability vectors of the
  // last ENSEMBLE_WINDOWS windows and decides on majority vote +
  // mean confidence + margin + debounce (see temporalEnsemble.ts).
  const ensembleRef = useRef<TemporalEnsemble | null>(null);
  // The last MAX_LIVE_FRAMES packed 345-float vectors —
  // the DTW second opinion scores these against the
  // canonical recordings when the ensemble emits from a
  // gray zone. SlidingWindowBuffer stores frames by
  // reference, so the same no-buffer-reuse contract
  // applies here.
  const liveFramesRef = useRef<Float32Array[]>([]);
  // Frame quality gate: drops frames with no usable hands and
  // resets the window when the signer stops signing, so a held
  // rest pose can neither pollute a window nor fire a
  // prediction, and the next sign starts from a clean buffer.
  const qualityGateRef = useRef(new FrameQualityGate());

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        if (DEBUG_INFERENCE) console.log('[useSignInference] Starting model load...');
        await tf.setBackend("webgl");
        await tf.ready();
        if (DEBUG_INFERENCE) console.log('[useSignInference] Backend ready, loading model...');

        const model = await tf.loadLayersModel(modelUrl);
        if (DEBUG_INFERENCE) console.log('[useSignInference] Model loaded successfully');
        if (cancelled) {
          model.dispose();
          return;
        }

        // Warm-up pass so the first real prediction isn't also paying for
        // WebGL shader compilation on the critical path (that first call
        // can otherwise take 10-100x longer than steady-state).
        const warm = tf.zeros([1, WINDOW_SIZE, FEATURE_DIM]);
        const warmOut = model.predict(warm) as tf.Tensor;
        warmOut.dispose();
        warm.dispose();

        modelRef.current = model;
        setReady(true);
      } catch (err) {
        if (!cancelled) {
          console.error('[useSignInference] Model load error:', err);
          setError(err instanceof Error ? err.message : "Failed to load model");
        }
      }
    })();

    return () => {
      cancelled = true;
      modelRef.current?.dispose();
      modelRef.current = null;
    };
  }, [modelUrl]);

  const pushFrame = useCallback(
    (frame: Float32Array) => {
      // Frame quality gate FIRST — a frame with no usable hands
      // would push a real sign's window straight into the rest
      // pose, and a held rest pose could otherwise fire a false
      // prediction from accumulated stale frames.
      const quality = qualityGateRef.current.evaluate(frame);
      pushCountRef.current++;
      if (quality.isRest) {
        restCountRef.current++;
        if (DEBUG_INFERENCE && pushCountRef.current % 30 === 0) {
          console.log(
            `[pipe] pushes=${pushCountRef.current} rest=${restCountRef.current} ` +
              `unusable=${unusableCountRef.current} emits=${emitCountRef.current}`
          );
        }
        // The signer stopped: clear all accumulated evidence so
        // the next sign is decided on fresh windows only.
        bufferRef.current.reset();
        ensembleRef.current?.reset();
        liveFramesRef.current.length = 0;
        return;
      }
      if (!quality.usable) {
        // Transient single-frame dropout — skip it rather than
        // corrupt the window, but keep the ensemble's evidence.
        unusableCountRef.current++;
        return;
      }

      const buffer = bufferRef.current;
      buffer.push(frame);
      const liveFrames = liveFramesRef.current;
      liveFrames.push(frame);
      if (liveFrames.length > MAX_LIVE_FRAMES) liveFrames.shift();

      const model = modelRef.current;
      if (!model || !buffer.shouldEmit()) return;

      if (
        !ensembleRef.current ||
        ensembleRef.current.classCount !== labels.length
      ) {
        ensembleRef.current = new TemporalEnsemble(labels.length, labels);
      }

      // tf.tidy synchronously frees every intermediate tensor created
      // inside the callback except whatever it returns. Without it, each
      // per-window predict() call below leaks a GPU-backed texture, and a
      // long inference session slowly exhausts WebGL memory.
      const probs = tf.tidy(() => {
        const input = tf.tensor(buffer.getWindow(), [1, WINDOW_SIZE, FEATURE_DIM]);
        return model.predict(input) as tf.Tensor;
      });

      probs
        .data()
        .then((data) => {
          // Bias correction: boost non-day labels (model over-predicts day labels).
          // Model trained on imbalanced data over-predicts day labels (MONDAY, TUESDAY, etc.).
          // Apply a 1.25x boost to all non-day labels to counter the bias.
          // 1.25x is a balanced boost to help other signs without false positives.
          const dayRegex = /\b(DAY|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY|TODAY|YESTERDAY)\b/;
          for (let i = 0; i < data.length; i++) {
            if (!dayRegex.test(labels[i])) {
              data[i] *= 1.25;
            }
          }
          if (DEBUG_INFERENCE) {
            const sorted = Array.from(data)
              .map((p, i) => ({ index: i, prob: p }))
              .sort((a, b) => b.prob - a.prob);
            const top3Str = sorted
              .slice(0, 3)
              .map(
                (entry) =>
                  `${EMISSION_ALIASES[labels[entry.index]] ?? labels[entry.index] ?? entry.index}(${entry.prob.toFixed(4)})`
              )
              .join(", ");
            console.log(`[predict] top3: ${top3Str}`);
          }
          // The ensemble owns the decision: ring-mean confidence,
          // top-1/top-2 margin, majority vote and emission debounce
          // all live in temporalEnsemble.ts. A single window — even a
          // confident one — can no longer fire a prediction on its own.
          const result = ensembleRef.current?.push(data as Float32Array);
          if (result) {
            const now = performance.now();
            const sinceLast = lastEmitTimeRef.current
              ? now - lastEmitTimeRef.current
              : 0;
            lastEmitTimeRef.current = now;
            emitCountRef.current++;
            if (DEBUG_INFERENCE) {
              console.log(
                `[pipe] emit #${emitCountRef.current} label=${result.label} ` +
                  `conf=${result.confidence.toFixed(3)} ` +
                  `margin=${(result.margin ?? 0).toFixed(3)} ` +
                  `runnerUp=${result.runnerUp ?? "n/a"} ` +
                  `ring=${ensembleRef.current?.size ?? 0} ` +
                  `sinceLastMs=${sinceLast.toFixed(0)} ` +
                  `pushes=${pushCountRef.current}`
              );
            }
            const emit = (
              label: string,
              confidence: number,
              secondOpinion?: {
                originalLabel: string;
                scores: Record<string, number>;
              }
            ) => {
              setPrediction({ label, confidence });
              // Measurement layer: every emission (with its
              // runner-up, and whether the second opinion
              // switched it) is logged on-device so accuracy
              // claims can be weighed against real usage
              // instead of guessed. See accuracyLog.ts.
              getAccuracyLogger().logEmission(
                performance.now(),
                label,
                confidence,
                result.runnerUp,
                secondOpinion
              );
            };

            const top3 = result.top3 ?? [];
            if (
              top3.length >= 2 &&
              needsSecondOpinion(result.confidence, result.margin ?? 0)
            ) {
              // Gray-zone emission: re-rank the ensemble's top
              // candidates against the canonical recordings with
              // the DTW engine before committing to a label.
              // Async (canonicals load on first use, cached
              // after) — but emissions are >= EMISSION_HOLD_MS
              // apart by design, so consults can't overtake
              // each other in practice.
              const candidates = top3;
              const live = liveFramesRef.current.slice();
withTimeout(
                  scoreCandidates(
                    candidates.map((c) => c.label),
                    live,
                    canonicalStore
                  ),
                  SECOND_OPINION_TIMEOUT_MS
                ).then((dtwScores) => {
                  if (!dtwScores || Object.keys(dtwScores).length === 0) {
                    if (DEBUG_INFERENCE) {
                      console.log(
                        `[pipe] second-opinion skipped (timeout or empty) ` +
                          `-> emit ensemble label=${result.label}`
                      );
                    }
                    emit(result.label, result.confidence);
                    return;
                  }
                const decision = applySecondOpinion(candidates, dtwScores);
                const finalLabel =
                  EMISSION_ALIASES[decision.label] ?? decision.label;
                emit(
                  finalLabel,
                  decision.confidence,
                  decision.switched
                    ? {
                        originalLabel:
                          EMISSION_ALIASES[decision.originalLabel] ??
                          decision.originalLabel,
                        scores: decision.scores,
                      }
                    : undefined
                );
              });
            } else {
              emit(result.label, result.confidence);
            }
          }
        })
        .finally(() => probs.dispose());
    },
    [labels]
  );

  return { ready, error, prediction, pushFrame };
}
