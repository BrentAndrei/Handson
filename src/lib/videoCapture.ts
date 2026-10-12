/**
 * Video → avatar-data capture engine.
 *
 * Turns a user-provided video file (e.g. an FSL tutorial clip) into the
 * exact JSON format served from /avatar-data/: flat frames of FEATURE_DIM
 * (345) floats — pose (33×3) + left hand (21×3) + right hand (21×3) +
 * the 40-point lip subset — each frame shoulder-midpoint normalized, the
 * same pipeline the live camera and the original dataset exporter use.
 *
 * Everything runs in the browser against the app's own MediaPipe WASM
 * build, so a captured clip is indistinguishable from a live capture:
 *   detectForVideo → packHolisticResult → normalizeLandmarks →
 *   selectFeatures.
 *
 * No server, no upload: the video never leaves the device.
 */
import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { packHolisticResult } from "../mediapipe/pack";
import { normalizeLandmarks } from "../utils/normalizeLandmarks";
import { selectFeatures, FEATURE_DIM } from "../features/featureSelect";
import { HAND_COUNT, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET } from "../mediapipe/types";

const MEDIAPIPE_VERSION = "0.10.20";
const MEDIAPIPE_WASM_BASE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const HOLISTIC_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

/** Per-group detection confidences. */
export interface CaptureConfidence {
  pose: number;
  face: number;
  hand: number;
}

/**
 * Detection confidence presets.
 *
 *   strict     — the live camera's thresholds (0.7). Cleanest landmarks,
 *                but drops hands on borderline frames.
 *   balanced   — 0.5 across the board. Default for tutorials.
 *   permissive — 0.3 pose/hand (the original dataset exporter's
 *                thresholds). Maximizes hand presence on clean,
 *                front-facing tutorial footage at some noise cost.
 */
export const CONFIDENCE_PRESETS: Record<string, CaptureConfidence> = {
  strict: { pose: 0.7, face: 0.7, hand: 0.7 },
  balanced: { pose: 0.5, face: 0.5, hand: 0.5 },
  permissive: { pose: 0.3, face: 0.5, hand: 0.3 },
};

export async function createVideoLandmarker(
  confidence: CaptureConfidence
): Promise<HolisticLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE_URL);
  return HolisticLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HOLISTIC_MODEL_URL,
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    minPoseDetectionConfidence: confidence.pose,
    minPosePresenceConfidence: confidence.pose,
    minFaceDetectionConfidence: confidence.face,
    minFacePresenceConfidence: confidence.face,
    minHandLandmarksConfidence: confidence.hand,
  });
}

export interface ClipRange {
  startSec: number;
  endSec: number;
}

/** Per-group presence over the extracted frames. */
export interface ClipCoverage {
  frames: number;
  pose: number;
  leftHand: number;
  rightHand: number;
  face: number;
}

export interface ExtractedClip {
  label: string;
  frameCount: number;
  frames: number[];
  coverage: ClipCoverage;
  fps: number;
}

const SEEK_TIMEOUT_MS = 400;

/**
 * Seek to a time and resolve once the frame is decodable.
 *
 * Some browsers do not fire `seeked` when the target equals the current
 * time, so a timeout backs the event up — extraction must never hang on
 * a no-op seek.
 */
function seekTo(video: HTMLVideoElement, timeSec: number): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;
    const timer = setTimeout(done, SEEK_TIMEOUT_MS);
    const onSeeked = () => done();
    function done() {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      video.removeEventListener("seeked", onSeeked);
      resolve();
    }
    video.addEventListener("seeked", onSeeked);
    video.currentTime = timeSec;
  });
}

export interface ExtractOptions {
  /** Sampling rate. The avatar-data loaders assume 30. */
  fps?: number;
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
}

/**
 * Extract one clip [startSec, endSec) at a fixed sample rate.
 *
 * Frames are seeked to individually and run through the real detection
 * pipeline — no frame is interpolated or synthesized, so the output is
 * ground-truth MediaPipe output for that footage, not a guess.
 */
export async function extractClip(
  video: HTMLVideoElement,
  landmarker: HolisticLandmarker,
  range: ClipRange,
  label: string,
  opts: ExtractOptions = {}
): Promise<ExtractedClip> {
  const fps = opts.fps ?? 30;
  const step = 1 / fps;

  const sampleTimes: number[] = [];
  for (let t = range.startSec; t < range.endSec - 1e-6; t += step) {
    sampleTimes.push(t);
  }
  if (sampleTimes.length === 0) {
    throw new Error("Clip range is empty — mark an in and out point.");
  }

  video.pause();

  const total = sampleTimes.length;
  const frames: number[] = new Array(total * FEATURE_DIM).fill(0);
  const coverage: ClipCoverage = {
    frames: total,
    pose: 0,
    leftHand: 0,
    rightHand: 0,
    face: 0,
  };

  // detectForVideo requires strictly increasing timestamps even when the
  // file position jumps backwards after a seek, so run a private clock
  // instead of passing wall-clock time.
  let clockMs = 0;
  const clockStepMs = 1000 / fps;
  let lastFeatures: Float32Array | null = null;

  for (let i = 0; i < total; i++) {
    if (opts.signal?.aborted) {
      throw new Error("Extraction cancelled");
    }

    await seekTo(video, sampleTimes[i]);

    let result: HolisticLandmarkerResult;
    try {
      result = landmarker.detectForVideo(video, clockMs);
    } catch {
      // A frame the model chokes on must not kill the clip: hold the
      // last good frame (the playback smoother tolerates held frames).
      clockMs += clockStepMs;
      if (lastFeatures) {
        for (let j = 0; j < FEATURE_DIM; j++) {
          frames[i * FEATURE_DIM + j] = lastFeatures[j];
        }
      }
      continue;
    }
    clockMs += clockStepMs;

    const packed = packHolisticResult(result, clockMs);
    const normalized = normalizeLandmarks(packed);
    const features = selectFeatures(normalized.vector);
    for (let j = 0; j < FEATURE_DIM; j++) {
      frames[i * FEATURE_DIM + j] = features[j];
    }
    lastFeatures = features;

    if (packed.hasPose) coverage.pose++;
    if (packed.hasLeftHand) coverage.leftHand++;
    if (packed.hasRightHand) coverage.rightHand++;
    if (packed.hasFace) coverage.face++;

    opts.onProgress?.(i + 1, total);
  }

  return { label, frameCount: total, frames, coverage, fps };
}

/**
 * avatar-data filename convention: the label with every non-alphanumeric
 * character replaced by "_" (the same sanitizing dataLoader applies when
 * it falls back from the manifest's explicit filename map).
 */
export function avatarDataFilename(label: string): string {
  return `${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`;
}

/** Serialize an extracted clip to the /avatar-data/ JSON shape. */
export function buildAvatarDataJson(clip: ExtractedClip): string {
  return JSON.stringify({
    label: clip.label,
    frameCount: clip.frameCount,
    frames: clip.frames,
  });
}

/**
 * MediaPipe pose landmark pairs that swap under a left/right mirror.
 * Eyes (1-6), ears (7/8), mouth (9/10), shoulders, elbows, wrists,
 * hand MCPs, hips, knees, ankles, heels, foot indices.
 */
const POSE_SYMMETRIC_PAIRS: [number, number][] = [
  [1, 4], [2, 5], [3, 6], [7, 8], [9, 10],
  [11, 12], [13, 14], [15, 16], [17, 18], [19, 20],
  [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32],
];

/**
 * Mirror a packed 345-float frame across the sagittal plane —
 * x is negated, the symmetric pose pairs swap, and the two hand
 * blocks exchange. Turns a LEFT-handed signer's recording into a
 * RIGHT-handed one (and vice versa) so the avatar always signs
 * with the user's dominant hand, regardless of who is in the
 * source video.
 *
 * The lip block (225..345) is left untouched: its mesh indices
 * are not trivially pairable, and no downstream consumer of the
 * avatar-data format reads it (the retargeter stops at 225, the
 * DTW scorer uses pose + hands only).
 *
 * An involution: mirrorFrame(mirrorFrame(x)) === x.
 */
export function mirrorFrame(
  src: Float32Array | number[],
  dst?: Float32Array
): Float32Array {
  const out = dst ?? new Float32Array(FEATURE_DIM);
  for (let i = 0; i < FEATURE_DIM; i += 3) {
    out[i] = -src[i];
    out[i + 1] = src[i + 1];
    out[i + 2] = src[i + 2];
  }
  for (const [a, b] of POSE_SYMMETRIC_PAIRS) {
    const ba = a * 3;
    const bb = b * 3;
    for (let c = 0; c < 3; c++) {
      const t = out[ba + c];
      out[ba + c] = out[bb + c];
      out[bb + c] = t;
    }
  }
  const handFloats = HAND_COUNT * 3;
  for (let i = 0; i < handFloats; i++) {
    const t = out[LEFT_HAND_OFFSET + i];
    out[LEFT_HAND_OFFSET + i] = out[RIGHT_HAND_OFFSET + i];
    out[RIGHT_HAND_OFFSET + i] = t;
  }
  return out;
}
