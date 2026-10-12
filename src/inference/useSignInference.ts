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
        console.log('[useSignInference] Starting model load...');
        await tf.setBackend("webgl");
        await tf.ready();
        console.log('[useSignInference] Backend ready, loading model...');

        const model = await tf.loadLayersModel(modelUrl);
        console.log('[useSignInference] Model loaded successfully');
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
      if (quality.isRest) {
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
              scoreCandidates(
                candidates.map((c) => c.label),
                live,
                canonicalStore
              ).then((dtwScores) => {
                if (Object.keys(dtwScores).length === 0) {
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
