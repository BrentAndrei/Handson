import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { FEATURE_DIM } from "../features/featureSelect";
import { DEFAULT_STRIDE, SlidingWindowBuffer, WINDOW_SIZE } from "../features/windowBuffer";

export interface Prediction {
  label: string;
  confidence: number;
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
  const stableLabelRef = useRef<string | null>(null);
  const stableCountRef = useRef(0);
  const PREDICTION_STABILITY_THRESHOLD = 2;

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
      const buffer = bufferRef.current;
      buffer.push(frame);

      const model = modelRef.current;
      if (!model || !buffer.shouldEmit()) return;

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
          const sorted = Array.from(data)
            .map((p, i) => ({ index: i, prob: p }))
            .sort((a, b) => b.prob - a.prob);
          const top3 = sorted.slice(0, 3);
          const top3Str = top3
            .map((entry) => `${labels[entry.index] ?? entry.index}(${entry.prob.toFixed(4)})`)
            .join(', ');
          console.log(`[predict] top3: ${top3Str}`);
          let bestIdx = sorted[0].index;
          const label = labels[bestIdx] ?? `class_${bestIdx}`;
          const confidence = data[bestIdx];
          if (label === stableLabelRef.current) {
            stableCountRef.current += 1;
          } else {
            stableLabelRef.current = label;
            stableCountRef.current = 1;
          }
          if (stableCountRef.current >= PREDICTION_STABILITY_THRESHOLD) {
            stableLabelRef.current = label;
            stableCountRef.current = 0;
            setPrediction({ label, confidence });
          }
        })
        .finally(() => probs.dispose());
    },
    [labels]
  );

  return { ready, error, prediction, pushFrame };
}
