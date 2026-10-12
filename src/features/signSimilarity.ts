/**
 * DTW sign-similarity — the practice-mode scorer.
 *
 * Compares a live signing attempt against a canonical recording of
 * the same sign and returns a 0..1 similarity with a per-part
 * breakdown (body / left hand / right hand). Pure maths, fully
 * client-side: no network, no model, no LLM.
 *
 * Both sides are reduced to a 49-value semantic feature vector per
 * frame — body directions and joint positions, the palm frame, and
 * per-finger curl — and aligned with Dynamic Time Warping, so
 * signing the same sign at a different speed still matches. DTW
 * compares shape, not timing.
 *
 * Spaces: live frames arrive as the shoulder-normalized 345-float
 * vector the avatar already receives (App.tsx handleLandmarks).
 * Canonical frames arrive raw from /avatar-data/<label>.json and are
 * normalized here with the same shoulder scheme, so both sides end
 * up in the same shoulder-width units. The normalization is
 * idempotent, so an already-normalized input passes through
 * unchanged. (The two sources genuinely differ today: the live path
 * normalizes at App.tsx:450, the recorded loader does not — this
 * module is the single place that puts them in one space.)
 *
 * The extractor deliberately does NOT call mediapipeToCanonicalPose:
 * that module keeps a module-level handedness hysteresis prior
 * (`lastHandAssignment`) shared with the avatar's own live calls,
 * and a second consumer would corrupt it. Everything computed here
 * is per-frame and stateless.
 */
import {
  LEFT_HAND_OFFSET,
  RIGHT_HAND_OFFSET,
} from "../mediapipe/types";

/** Floats per packed MediaPipe frame (pose 33 + hands 21×2 + lip subset 40). */
const FRAME_FLOATS = 345;
const POSE_OFFSET = 0;

/** Pose landmark indices used by the feature extractor. */
const NOSE = 0;
const L_SHOULDER = 11;
const R_SHOULDER = 12;
const L_ELBOW = 13;
const R_ELBOW = 14;
const L_WRIST = 15;
const R_WRIST = 16;
const L_HIP = 23;
const R_HIP = 24;

/** Finger base landmark indices: [thumb, index, middle, ring, pinky] MCPs. */
const FINGER_MCP = [1, 5, 9, 13, 17] as const;

/**
 * Feature vector layout (49 values):
 *   [0..27)  body: spineDir, shoulderAxis, L/R shoulder/elbow/wrist
 *            positions, headDir
 *   [27..38) left hand: palmDir(3), palmNormal(3), 5 finger curls
 *   [38..49) right hand: same
 */
export const FEATURE_DIM = 49;
const BODY_DIM = 27;
const HAND_DIM = 11;

/**
 * Finger curls weigh double: handshape is the most distinctive
 * signal in FSL (the same arm trajectory with a different handshape
 * is a different sign), so a curl mismatch must outscore a position
 * mismatch of the same magnitude. Tunable.
 */
const CURL_WEIGHT = 2.0;

const FEATURE_WEIGHTS = new Float32Array(FEATURE_DIM).fill(1);
for (let i = BODY_DIM + 6; i < BODY_DIM + HAND_DIM; i++) {
  FEATURE_WEIGHTS[i] = CURL_WEIGHT;
}
for (let i = BODY_DIM + HAND_DIM + 6; i < FEATURE_DIM; i++) {
  FEATURE_WEIGHTS[i] = CURL_WEIGHT;
}

export interface SignSimilarityConfig {
  /**
   * Mean aligned per-frame distance (in shoulder-widths) that maps
   * to a score of e^-1 ≈ 0.37. A mean distance of ~0.15 (a solid
   * same-sign performance) scores ~0.74; ~0.5 (a different sign)
   * scores ~0.37. Starting point — calibrate against recorded
   * pairs once practice-session data exists.
   */
  distanceScale: number;
  /** Minimum live frames before an attempt can be scored. */
  minLiveFrames: number;
}

export const DEFAULT_SIMILARITY_CONFIG: SignSimilarityConfig = {
  distanceScale: 0.5,
  minLiveFrames: 8,
};

export interface SignSimilarityResult {
  label: string;
  /** Overall similarity 0..1 (exp-decayed mean DTW distance). */
  score: number;
  /** Body-only similarity 0..1. */
  body: number;
  /** Left-hand-only similarity 0..1. */
  leftHand: number;
  /** Right-hand-only similarity 0..1. */
  rightHand: number;
  /** Mean aligned per-frame distance, shoulder-widths. */
  meanDistance: number;
  liveFrames: number;
  canonicalFrames: number;
}

/** Read landmark `index` from a packed group into a scratch triple. */
function readPoint(
  buf: Float32Array,
  groupOffset: number,
  index: number,
  out: number[]
): number[] {
  const base = groupOffset + index * 3;
  out[0] = buf[base];
  out[1] = buf[base + 1];
  out[2] = buf[base + 2];
  return out;
}

function subInto(out: number[], a: number[], b: number[]): number[] {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
  return out;
}

function crossInto(out: number[], a: number[], b: number[]): number[] {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x;
  out[1] = y;
  out[2] = z;
  return out;
}

function normalizeInto(out: number[]): number[] {
  const len = Math.sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
  if (len < 1e-10) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    return out;
  }
  out[0] /= len;
  out[1] /= len;
  out[2] /= len;
  return out;
}

/**
 * Per-finger curl from the PIP bend angle, using the same remap as
 * the avatar's computeFingerPose (PHASE 28): the raw PIP-only bend
 * is (1 - dot)/2, which reaches 0.5 at a 90° bend, so dividing by
 * 0.5 maps a closed fist to curl 1.0 and a straight finger to 0.
 * Angles are invariant under the MediaPipe→canonical affine map, so
 * this can run directly on packed (unconverted) landmarks.
 */
function fingerCurl(
  buf: Float32Array,
  groupOffset: number,
  mcpIndex: number
): number {
  const mcp = [0, 0, 0];
  const pip = [0, 0, 0];
  const dip = [0, 0, 0];
  readPoint(buf, groupOffset, mcpIndex, mcp);
  readPoint(buf, groupOffset, mcpIndex + 1, pip);
  readPoint(buf, groupOffset, mcpIndex + 2, dip);

  const mcpToPip = subInto([0, 0, 0], pip, mcp);
  const pipToDip = subInto([0, 0, 0], dip, pip);
  normalizeInto(mcpToPip);
  normalizeInto(pipToDip);

  const dot =
    mcpToPip[0] * pipToDip[0] +
    mcpToPip[1] * pipToDip[1] +
    mcpToPip[2] * pipToDip[2];
  const rawCurl = (1 - dot) / 2;
  return Math.max(0, Math.min(1, rawCurl / 0.5));
}

/**
 * Reduce one packed 345-float frame to its 49-value feature vector.
 * `out` must have length FEATURE_DIM.
 */
export function frameToFeatureVector(
  buf: Float32Array,
  out: Float32Array = new Float32Array(FEATURE_DIM)
): Float32Array {
  // --- Body (27 values) ---
  const lShoulder = [0, 0, 0];
  const rShoulder = [0, 0, 0];
  const lElbow = [0, 0, 0];
  const rElbow = [0, 0, 0];
  const lWrist = [0, 0, 0];
  const rWrist = [0, 0, 0];
  const lHip = [0, 0, 0];
  const rHip = [0, 0, 0];
  const nose = [0, 0, 0];
  readPoint(buf, POSE_OFFSET, L_SHOULDER, lShoulder);
  readPoint(buf, POSE_OFFSET, R_SHOULDER, rShoulder);
  readPoint(buf, POSE_OFFSET, L_ELBOW, lElbow);
  readPoint(buf, POSE_OFFSET, R_ELBOW, rElbow);
  readPoint(buf, POSE_OFFSET, L_WRIST, lWrist);
  readPoint(buf, POSE_OFFSET, R_WRIST, rWrist);
  readPoint(buf, POSE_OFFSET, L_HIP, lHip);
  readPoint(buf, POSE_OFFSET, R_HIP, rHip);
  readPoint(buf, POSE_OFFSET, NOSE, nose);

  const shoulderMid = [
    (lShoulder[0] + rShoulder[0]) / 2,
    (lShoulder[1] + rShoulder[1]) / 2,
    (lShoulder[2] + rShoulder[2]) / 2,
  ];
  const hipMid = [
    (lHip[0] + rHip[0]) / 2,
    (lHip[1] + rHip[1]) / 2,
    (lHip[2] + rHip[2]) / 2,
  ];

   // spineDir: shoulder midpoint → away from hips.
   //
   // All three body directions below are confined to the FRONTAL
   // PLANE (z dropped). The corpus was captured with an angled rig
   // (training/verify_capture_quality.py: the nose sits 0.7-1.1
   // torso-lengths forward of the shoulder line; a frontal capture
   // measures 0.10-0.15). Measured across the recordings in
   // shoulder-width units after normalization (scripts/
   // measure-rig-distortion.ts): headDir z = -0.925, stable to
   // 0.017 frame-to-frame on 100% of frames; spineDir z = -0.364
   // with 0.158 jitter — while a live, front-facing student
   // measures ~-0.15. Kept in 3D, that systematic offset alone
   // produced a per-frame headDir distance of ~1.0 shoulder-widths,
   // capping a PERFECT attempt's body score near 0.12
   // (exp(-1.0/0.5)). The avatar's adapter guards the same defect
   // (computeBodyPose PHASE 21 confines the trunk/head axis to
   // the frontal plane); this is the matching guard for the scorer,
   // and it makes the comparison invariant to the capture rig.
   // The in-plane (x, y) components are the real signal — trunk
   // lean, shoulder-line tilt, head tilt — and are kept, as are
   // the joint POSITIONS above (hand depth is genuine motion
   // signal, and the rig offset cancels between shoulders and
   // wrists anyway).
   const spineRaw = subInto([0, 0, 0], shoulderMid, hipMid);
   const spineDir = normalizeInto([spineRaw[0], spineRaw[1], 0]);
   // shoulderAxis: left → right shoulder.
   const axisRaw = subInto([0, 0, 0], rShoulder, lShoulder);
   const shoulderAxis = normalizeInto([axisRaw[0], axisRaw[1], 0]);
   // headDir: shoulder midpoint → nose.
   const headRaw = subInto([0, 0, 0], nose, shoulderMid);
   const headDir = normalizeInto([headRaw[0], headRaw[1], 0]);

  out[0] = spineDir[0];
  out[1] = spineDir[1];
  out[2] = spineDir[2];
  out[3] = shoulderAxis[0];
  out[4] = shoulderAxis[1];
  out[5] = shoulderAxis[2];
  out[6] = lShoulder[0];
  out[7] = lShoulder[1];
  out[8] = lShoulder[2];
  out[9] = lElbow[0];
  out[10] = lElbow[1];
  out[11] = lElbow[2];
  out[12] = lWrist[0];
  out[13] = lWrist[1];
  out[14] = lWrist[2];
  out[15] = rShoulder[0];
  out[16] = rShoulder[1];
  out[17] = rShoulder[2];
  out[18] = rElbow[0];
  out[19] = rElbow[1];
  out[20] = rElbow[2];
  out[21] = rWrist[0];
  out[22] = rWrist[1];
  out[23] = rWrist[2];
  out[24] = headDir[0];
  out[25] = headDir[1];
  out[26] = headDir[2];

  // --- Hands (11 values each: palmDir 3, palmNormal 3, curls 5) ---
  extractHand(buf, LEFT_HAND_OFFSET, out, BODY_DIM, "left");
  extractHand(buf, RIGHT_HAND_OFFSET, out, BODY_DIM + HAND_DIM, "right");

  return out;
}

function extractHand(
  buf: Float32Array,
  groupOffset: number,
  out: Float32Array,
  base: number,
  side: "left" | "right"
): void {
  const wrist = [0, 0, 0];
  const indexMcp = [0, 0, 0];
  const pinkyMcp = [0, 0, 0];
  const middleMcp = [0, 0, 0];
  readPoint(buf, groupOffset, 0, wrist);
  readPoint(buf, groupOffset, 5, indexMcp);
  readPoint(buf, groupOffset, 9, middleMcp);
  readPoint(buf, groupOffset, 17, pinkyMcp);

  // palmDir: wrist → middle MCP (matches computeHandPose).
  const palmDir = normalizeInto(subInto([0, 0, 0], middleMcp, wrist));

  // Palm normal: the operand order depends on the hand, exactly as
  // in computeHandPose (PHASE 15) — using the wrong order flips the
  // normal 180° on one side and makes every left/right comparison
  // score as a mismatch.
  const toIndex = subInto([0, 0, 0], indexMcp, wrist);
  const toPinky = subInto([0, 0, 0], pinkyMcp, wrist);
  const palmNormal =
    side === "right"
      ? normalizeInto(crossInto([0, 0, 0], toPinky, toIndex))
      : normalizeInto(crossInto([0, 0, 0], toIndex, toPinky));

  out[base] = palmDir[0];
  out[base + 1] = palmDir[1];
  out[base + 2] = palmDir[2];
  out[base + 3] = palmNormal[0];
  out[base + 4] = palmNormal[1];
  out[base + 5] = palmNormal[2];
  for (let f = 0; f < FINGER_MCP.length; f++) {
    out[base + 6 + f] = fingerCurl(buf, groupOffset, FINGER_MCP[f]);
  }
}

/**
 * Shoulder-midpoint + shoulder-width normalization for a packed
 * 345-float vector — the same scheme as normalizeLandmarks, applied
 * directly to the packed buffer. Idempotent: an already-normalized
 * vector has anchor ≈ 0 and scale ≈ 1, so it passes through
 * unchanged. Mirrors normalizeLandmarks' degenerate guards.
 */
export function normalizePackedVector(
  buf: Float32Array,
  out: Float32Array = new Float32Array(buf.length)
): Float32Array {
  const lShoulderBase = POSE_OFFSET + L_SHOULDER * 3;
  const rShoulderBase = POSE_OFFSET + R_SHOULDER * 3;
  const ax = (buf[lShoulderBase] + buf[rShoulderBase]) / 2;
  const ay = (buf[lShoulderBase + 1] + buf[rShoulderBase + 1]) / 2;
  const az = (buf[lShoulderBase + 2] + buf[rShoulderBase + 2]) / 2;

  const dx = buf[rShoulderBase] - buf[lShoulderBase];
  const dy = buf[rShoulderBase + 1] - buf[lShoulderBase + 1];
  const dz = buf[rShoulderBase + 2] - buf[lShoulderBase + 2];
  let scale = Math.sqrt(dx * dx + dy * dy + dz * dz);

  if (scale <= 1e-4) {
    // Degenerate pose estimate: fall back to the centroid + bounding
    // box diagonal of whatever landmarks are present (same fallback
    // as normalizeLandmarks), so absent-pose frames cannot produce
    // NaN. Present groups only — zero-filled groups are skipped.
    let sx = 0, sy = 0, sz = 0, n = 0;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let p = 0; p < FRAME_FLOATS / 3; p++) {
      const b = p * 3;
      const x = buf[b], y = buf[b + 1], z = buf[b + 2];
      if (x === 0 && y === 0 && z === 0) continue;
      sx += x; sy += y; sz += z;
      n++;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    if (n > 0) {
      out.set(buf);
      const cx = sx / n, cy = sy / n, cz = sz / n;
      const diag = Math.sqrt(
        (maxX - minX) ** 2 + (maxY - minY) ** 2 + (maxZ - minZ) ** 2
      );
      const s = diag > 1e-4 ? diag : 1;
      for (let i = 0; i < out.length; i += 3) {
        out[i] = (out[i] - cx) / s;
        out[i + 1] = (out[i + 1] - cy) / s;
        out[i + 2] = (out[i + 2] - cz) / s;
      }
      return out;
    }
    out.set(buf);
    return out;
  }

  for (let i = 0; i < out.length; i += 3) {
    out[i] = (buf[i] - ax) / scale;
    out[i + 1] = (buf[i + 1] - ay) / scale;
    out[i + 2] = (buf[i + 2] - az) / scale;
  }
  return out;
}

/** Weighted Euclidean distance between two feature vectors. */
function frameDistance(
  a: Float32Array,
  b: Float32Array,
  weights: Float32Array
): number {
  let sum = 0;
  for (let k = 0; k < a.length; k++) {
    const d = a[k] - b[k];
    sum += weights[k] * d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Dynamic Time Warping mean distance between two feature sequences.
 *
 * Classic cumulative-cost DTW with the symmetric step pattern,
 * normalized by (n + m) so sequences of different lengths are
 * comparable. Returns Infinity for empty input. O(n·m) time, O(m)
 * memory (two rolling rows).
 */
export function dtwMeanDistance(
  a: Float32Array[],
  b: Float32Array[],
  weights: Float32Array = FEATURE_WEIGHTS
): number {
  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return Infinity;

  let prev = new Float64Array(m);
  let curr = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    const ai = a[i];
    for (let j = 0; j < m; j++) {
      const d = frameDistance(ai, b[j], weights);
      let best: number;
      if (i === 0 && j === 0) {
        best = d;
      } else if (i === 0) {
        best = curr[j - 1] + d;
      } else if (j === 0) {
        best = prev[j] + d;
      } else {
        best = Math.min(prev[j], curr[j - 1], prev[j - 1]) + d;
      }
      curr[j] = best;
    }
    const swap = prev;
    prev = curr;
    curr = swap;
  }
  return prev[m - 1] / (n + m);
}

function scoreFromDistance(meanDistance: number, config: SignSimilarityConfig): number {
  if (!Number.isFinite(meanDistance)) return 0;
  return Math.exp(-meanDistance / config.distanceScale);
}

/**
 * Load a sign's canonical recording as raw packed frames (the same
 * 345-float frames /avatar-data/ serves the avatar loader). Returns
 * null when the label has no recording. Frames are NOT smoothed
 * here: the reference should be the ground-truth capture, not the
 * avatar's playback-smoothed variant.
 */
export async function loadCanonicalRawFrames(
  label: string
): Promise<Float32Array[] | null> {
  const sanitized = label.replace(/[^a-zA-Z0-9]/g, "_");
  try {
    const resp = await fetch(`/avatar-data/${sanitized}.json`);
    if (!resp.ok) return null;
    const data = (await resp.json()) as { frames?: unknown };
    if (!data || !Array.isArray(data.frames) || data.frames.length === 0) {
      return null;
    }
    const flat = new Float32Array(data.frames as unknown as number[]);
    const frameCount = Math.floor(flat.length / FRAME_FLOATS);
    if (frameCount === 0) return null;
    const frames: Float32Array[] = [];
    for (let i = 0; i < frameCount; i++) {
      frames.push(flat.slice(i * FRAME_FLOATS, (i + 1) * FRAME_FLOATS));
    }
    return frames;
  } catch {
    return null;
  }
}

/**
 * Stateful scorer for one practice attempt against one sign.
 *
 * Feed it the live normalized vector each frame (the exact object
 * App.tsx hands the avatar), then finish() for the result.
 */
export class SignSimilarityTracker {
  private readonly canonical: Float32Array[] = [];
  private readonly live: Float32Array[] = [];
  private readonly scratch = new Float32Array(FEATURE_DIM);

  constructor(
    label: string,
    canonicalRawFrames: Float32Array[],
    private readonly config: SignSimilarityConfig = DEFAULT_SIMILARITY_CONFIG
  ) {
    this.label = label;
    const vec = new Float32Array(FRAME_FLOATS);
    for (const raw of canonicalRawFrames) {
      if (raw.length !== FRAME_FLOATS) continue;
      normalizePackedVector(raw, vec);
      const feats = frameToFeatureVector(vec, new Float32Array(FEATURE_DIM));
      if (isFiniteVector(feats)) this.canonical.push(feats);
    }
  }

  private readonly label: string;

  pushLiveFrame(normalizedVector: Float32Array): void {
    if (normalizedVector.length !== FRAME_FLOATS) return;
    const feats = frameToFeatureVector(normalizedVector, this.scratch);
    if (!isFiniteVector(feats)) return;
    // copy out of the reusable scratch
    this.live.push(new Float32Array(feats));
  }

  finish(): SignSimilarityResult | null {
    if (this.live.length < this.config.minLiveFrames || this.canonical.length === 0) {
      return null;
    }

    const overall = dtwMeanDistance(this.live, this.canonical);
    const bodyLive = slicePart(this.live, 0, BODY_DIM);
    const bodyCanon = slicePart(this.canonical, 0, BODY_DIM);
    const lHandLive = slicePart(this.live, BODY_DIM, BODY_DIM + HAND_DIM);
    const lHandCanon = slicePart(this.canonical, BODY_DIM, BODY_DIM + HAND_DIM);
    const rHandLive = slicePart(this.live, BODY_DIM + HAND_DIM, FEATURE_DIM);
    const rHandCanon = slicePart(
      this.canonical,
      BODY_DIM + HAND_DIM,
      FEATURE_DIM
    );
    const wBody = FEATURE_WEIGHTS.slice(0, BODY_DIM);
    const wLHand = FEATURE_WEIGHTS.slice(BODY_DIM, BODY_DIM + HAND_DIM);
    const wRHand = FEATURE_WEIGHTS.slice(BODY_DIM + HAND_DIM, FEATURE_DIM);

    const bodyDist = dtwMeanDistance(bodyLive, bodyCanon, wBody);
    const lDist = dtwMeanDistance(lHandLive, lHandCanon, wLHand);
    const rDist = dtwMeanDistance(rHandLive, rHandCanon, wRHand);

    return {
      label: this.label,
      score: scoreFromDistance(overall, this.config),
      body: scoreFromDistance(bodyDist, this.config),
      leftHand: scoreFromDistance(lDist, this.config),
      rightHand: scoreFromDistance(rDist, this.config),
      meanDistance: Number.isFinite(overall) ? overall : Infinity,
      liveFrames: this.live.length,
      canonicalFrames: this.canonical.length,
    };
  }
}

function slicePart(
  vecs: Float32Array[],
  start: number,
  end: number
): Float32Array[] {
  const out: Float32Array[] = new Array(vecs.length);
  for (let i = 0; i < vecs.length; i++) {
    out[i] = vecs[i].slice(start, end);
  }
  return out;
}

function isFiniteVector(v: Float32Array): boolean {
  for (let i = 0; i < v.length; i++) {
    if (!Number.isFinite(v[i])) return false;
  }
  return true;
}
