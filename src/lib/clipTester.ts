/**
 * Offline clip → classifier test.
 *
 * The capture studio's verification step: before a captured clip
 * is installed as avatar-data, run it through the SAME tfjs model
 * the live Translate view uses and report what the model reads.
 * A clip the model disagrees with is either a bad capture (hands
 * dropped, wrong framing) or a genuinely new sign — the verdict
 * tells the user which, before the clip reaches the dataset.
 *
 * Method: slice the clip into the same [WINDOW_SIZE, FEATURE_DIM]
 * sliding windows the live pipeline feeds the model (same stride),
 * predict each window, and average the output vectors. The mean is
 * ranked against INFERENCE_LABELS — the exact order the model's
 * output layer was trained with.
 */
import * as tf from "@tensorflow/tfjs";
import { FEATURE_DIM } from "../features/featureSelect";
import {
  DEFAULT_STRIDE,
  WINDOW_SIZE,
  sliceIntoWindows,
} from "../features/windowBuffer";

/** Same URL the live inference hook uses (App.tsx). */
export const SIGN_MODEL_URL = "/models/fsl_model_v2/model.json";
export const SIGN_MODEL_CLASSES = 303;

let modelPromise: Promise<tf.LayersModel> | null = null;

/**
 * Lazily loads the sign classifier once per page. A failed load
 * does not poison the singleton — the next call retries.
 */
export function loadSignModel(): Promise<tf.LayersModel> {
  if (!modelPromise) {
    modelPromise = (async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      return tf.loadLayersModel(SIGN_MODEL_URL);
    })();
    modelPromise.catch(() => {
      modelPromise = null;
    });
  }
  return modelPromise;
}

export interface ClipPrediction {
  label: string;
  /** Mean probability across the clip's windows (0..1). */
  confidence: number;
}

export interface ClipTestResult {
  /** Top predictions, descending confidence. */
  top: ClipPrediction[];
  /** True when the clip's own label is the top prediction. */
  agree: boolean;
  /** How many sliding windows the clip produced. */
  windows: number;
}

/**
 * Test one captured clip against the classifier.
 *
 * `flatFrames` is the avatar-data payload: frameCount × FEATURE_DIM
 * floats. Throws when the clip is shorter than one window — there
 * is not enough temporal context for the model to say anything.
 */
export async function testClipAgainstModel(
  model: tf.LayersModel,
  flatFrames: number[],
  label: string,
  labels: string[],
  onProgress?: (done: number, total: number) => void
): Promise<ClipTestResult> {
  const frameCount = Math.floor(flatFrames.length / FEATURE_DIM);
  const frames: Float32Array[] = [];
  for (let i = 0; i < frameCount; i++) {
    frames.push(
      new Float32Array(
        flatFrames.slice(i * FEATURE_DIM, (i + 1) * FEATURE_DIM)
      )
    );
  }

  const windows = sliceIntoWindows(
    frames,
    FEATURE_DIM,
    WINDOW_SIZE,
    DEFAULT_STRIDE
  );
  if (windows.length === 0) {
    throw new Error(
      `Clip too short to test — need at least ${WINDOW_SIZE} frames, got ${frameCount}.`
    );
  }

  const sums = new Float64Array(labels.length);
  for (let w = 0; w < windows.length; w++) {
    // tf.tidy frees every intermediate tensor the predict() call
    // creates; only the returned tensor survives to .data().
    const probs = tf.tidy(() => {
      const input = tf.tensor(windows[w], [1, WINDOW_SIZE, FEATURE_DIM]);
      return model.predict(input) as tf.Tensor;
    });
    try {
      const data = await probs.data();
      const classCount = Math.min(data.length, labels.length);
      for (let c = 0; c < classCount; c++) {
        sums[c] += data[c];
      }
    } finally {
      probs.dispose();
    }
    onProgress?.(w + 1, windows.length);
  }

  const top: ClipPrediction[] = Array.from(sums)
    .map((sum, i) => ({
      label: labels[i],
      confidence: sum / windows.length,
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  return {
    top,
    agree: top.length > 0 && top[0].label === label,
    windows: windows.length,
  };
}
