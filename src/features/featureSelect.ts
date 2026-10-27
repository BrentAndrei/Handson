import { RIGHT_HAND_OFFSET } from "../mediapipe/types";

/**
 * Extracts only the right hand landmarks (21 points * 3 = 63 features).
 * The retrained BiGRU (fsl_bigru_v2_kaggle) was trained on Kaggle FSL
 * dataset which provides single-hand (dominant/right hand) landmarks.
 * 21 landmarks * 3 (x,y,z) = 63 features per frame.
 */
export const FEATURE_DIM = 21 * 3; // 63

/**
 * Reduces a full normalized holistic vector down to right-hand-only
 * landmarks (21 points * 3 coords = 63 features). This matches the
 * Kaggle FSL dataset format used to train fsl_bigru_v2_kaggle.
 */
export function selectFeatures(vector: Float32Array): Float32Array {
  const out = new Float32Array(63);
  out.set(
    new Float32Array(vector.buffer, RIGHT_HAND_OFFSET * 4, 63 * 4)
  );
  return out;
}