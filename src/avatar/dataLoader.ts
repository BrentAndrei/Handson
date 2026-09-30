import { retargetFrames, smoothFramesSpline, type AvatarFrame, type ParentWorldQuats } from './retarget';
import { smplxToAvatarFrames, type SmplxFrame, type SmplxPose } from './smplxConverter';
import { smplxPoseToCanonicalPose } from './smplxAdapter';
import { mediapipeToCanonicalPose } from './mediapipeAdapter';
import { LandmarkSmoother } from '../utils/landmarkSmoother';
import type { CanonicalPose } from './canonicalPose';

/** Floats per MediaPipe Holistic frame: 115 landmarks x 3. */
const MEDIAPIPE_FRAME_FLOATS = 345;

interface AvatarData {
  label: string;
  frameCount: number;
  frames: number[][];
}

interface AvatarDataset {
  label: string;
  frames: AvatarFrame[];
  smoothed: AvatarFrame[];
}

/** Dataset using the new CanonicalPose pipeline. */
interface CanonicalAvatarDataset {
  label: string;
  frames: (CanonicalPose | null)[];
}

const dataCache: Map<string, AvatarDataset> = new Map();
const canonicalCache: Map<string, CanonicalAvatarDataset> = new Map();
const mediaPipeCache: Map<string, CanonicalAvatarDataset> = new Map();
let allAvailableLabels: string[] = [];
let labelsLoaded = false;

// Label-to-filename lookup maps, built from manifests
let sourceFileMap: Map<string, string> = new Map(); // label -> source filename (e.g. "HE/SHE" -> "HE_SHE.json")
let smplxFileByLabel: Map<string, string> = new Map(); // label -> smplx filename (e.g. "HE/SHE" -> "HE_SHE.smplx.json")
let smplxFileMap: Map<string, string> = new Map();  // label -> smplx filename from SMPL-X manifest entries

// SMPL-X service configuration
const SMPLX_SERVICE_URL = (import.meta.env.VITE_SMPLX_SERVICE_URL as string) ||
  (typeof window !== 'undefined' && (window as any).__smplxServiceUrl) ||
  'http://localhost:8000';

export function isSmplxServiceAvailable(): boolean {
  return typeof window !== 'undefined' && (window as any).__useSmplxService === true;
}

export function setSmplxServiceUrl(url: string): void {
  if (typeof window !== 'undefined') {
    (window as any).__smplxServiceUrl = url;
  }
}

export function enableSmplxService(enabled: boolean): void {
  if (typeof window !== 'undefined') {
    (window as any).__useSmplxService = enabled;
  }
}

export async function loadAvatarData(): Promise<string[]> {
  if (labelsLoaded) return allAvailableLabels;

  try {
    const manifestResp = await fetch('/avatar-data/manifest.json');
    if (!manifestResp.ok) throw new Error('Manifest not found');
    const manifest = await manifestResp.json();
    const labels = manifest.labels || [];
    if (labels.length === 0) {
      throw new Error('Manifest has no labels');
    }

    // Build label-to-file map from entries if available, or derive from labels
    if (manifest.entries && manifest.entries.length > 0) {
      for (const entry of manifest.entries) {
        if (entry.label && entry.file) {
          sourceFileMap.set(entry.label, entry.file);
        }
        if (entry.label && entry.smplxFile) {
          smplxFileByLabel.set(entry.label, entry.smplxFile);
        }
      }
    } else {
      // Derive filenames from labels (files use underscore format)
      for (const label of labels) {
        const sanitized = label.replace(/[^a-zA-Z0-9]/g, '_');
        sourceFileMap.set(label, `${sanitized}.json`);
        smplxFileByLabel.set(label, `${sanitized}.smplx.json`);
      }
    }

    allAvailableLabels = labels;
    labelsLoaded = true;

    // Also load SMPL-X manifest to build the file lookup map
    try {
      const smplxManifestResp = await fetch('/avatar-data-smplx/manifest.json', { signal: AbortSignal.timeout(3000) });
      if (smplxManifestResp.ok) {
        const smplxManifest = await smplxManifestResp.json();
        if (smplxManifest.entries && smplxManifest.entries.length > 0) {
          for (const entry of smplxManifest.entries) {
            if (entry.label && entry.file) {
              smplxFileMap.set(entry.label, entry.file);
            }
          }
        }
      }
    } catch (e) {
      console.warn('[dataLoader] SMPL-X manifest load failed:', e);
    }

    return allAvailableLabels;
  } catch {
    labelsLoaded = true;
    return [];
  }
}

export function resetAvatarData(): void {
  dataCache.clear();
  allAvailableLabels = [];
  labelsLoaded = false;
  sourceFileMap.clear();
  smplxFileByLabel.clear();
  smplxFileMap.clear();
}

export async function loadLabelData(label: string, parentWorldQuats?: ParentWorldQuats): Promise<AvatarDataset | null> {
  const cached = dataCache.get(label);
  if (cached) return cached;

  // Try SMPL-X path if the service is enabled
  if (isSmplxServiceAvailable()) {
    const smplxData = await loadLabelDataSmplx(label, parentWorldQuats);
    if (smplxData) {
      dataCache.set(label, smplxData);
      return smplxData;
    }
  }

  // Fallback: traditional quaternion-based retargeting from MediaPipe landmarks
  const sanitized = label.replace(/[^a-zA-Z0-9]/g, '_');
  // Look up the actual filename from the manifest map; fall back to sanitized
  const sourceFile = sourceFileMap.get(label) || `${sanitized}.json`;
  const url = `/avatar-data/${sourceFile}`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const data: AvatarData = await resp.json();

    const dataArray = new Float32Array(data.frames.flat());
    const rawFrames = retargetFrames(dataArray, parentWorldQuats);
    const smoothed = smoothFramesSpline(rawFrames, 5);

    const dataset: AvatarDataset = {
      label: data.label,
      frames: smoothed,
      smoothed,
    };

    dataCache.set(label, dataset);
    return dataset;
  } catch {
    return null;
  }
}

/** Load avatar data from the SMPL-X service (hybrid approach). */
async function loadLabelDataSmplx(
  label: string,
  parentWorldQuats?: ParentWorldQuats,
): Promise<AvatarDataset | null> {
  try {
    // First try pre-computed SMPL-X data
    const sanitized = label.replace(/[^a-zA-Z0-9]/g, '_');
    const smplxUrl = `${SMPLX_SERVICE_URL}/avatar-data-smplx/${sanitized}.json`;

    let smplxData: SmplxFrame | null = null;

    // 1. Try local pre-computed SMPL-X data (served from public/)
    try {
      // Look up the actual SMPL-X filename from the source manifest's smplxFile field
      const localFile = smplxFileByLabel.get(label) || smplxFileMap.get(label) || `${sanitized}.smplx.json`;
      const localUrl = `/avatar-data-smplx/${localFile}`;
      const resp = await fetch(localUrl, { signal: AbortSignal.timeout(3000) });
      if (resp.ok) {
        smplxData = await resp.json();
      }
    } catch {
    }

    // 2. Fall back to SMPL-X service for pre-computed data
    if (!smplxData) {
      try {
        const resp = await fetch(smplxUrl, { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
          smplxData = await resp.json();
        }
      } catch {
      }
    }

    if (!smplxData) {
      // No SMPL-X data available (local or service).
      // Return null and let the caller handle the fallback to quaternion
      // retargeting, which works without any service.
      return null;
    }

    const parentQuatDict = parentWorldQuats
      ? Object.fromEntries(
          Object.entries(parentWorldQuats).map(([k, v]) => [k, v as [number, number, number, number]]),
        )
      : undefined;

    const avatarFrames = smplxToAvatarFrames(smplxData.frames || [], parentQuatDict);
    const smoothed = smoothFramesSpline(avatarFrames, 5);

    return {
      label: smplxData.label || label,
      frames: smoothed,
      smoothed,
    };
  } catch (e) {
    console.error('[dataLoader] SMPL-X load failed, falling back:', e);
    return null;
  }
}

/**
 * Load SMPL-X data as CanonicalPose[] (new pipeline).
 *
 * This converts each SmplxPose → CanonicalPose using the new smplxAdapter,
 * which reads joint positions from `smplx_joints` (128×3 floats).
 * The resulting CanonicalPose[] is consumed by retargeter.ts.
 */
export async function loadLabelDataCanonical(
  label: string
): Promise<CanonicalAvatarDataset | null> {
  const cached = canonicalCache.get(label);
  if (cached) return cached;

  try {
    const sanitized = label.replace(/[^a-zA-Z0-9]/g, '_');
    const localFile = smplxFileByLabel.get(label) || smplxFileMap.get(label) || `${sanitized}.smplx.json`;
    const localUrl = `/avatar-data-smplx/${localFile}`;

    let smplxData: SmplxFrame | null = null;

    try {
      const resp = await fetch(localUrl, { signal: AbortSignal.timeout(3000) });
      if (resp.ok) {
        smplxData = await resp.json();
      }
    } catch {}

    if (!smplxData) {
      try {
        const resp = await fetch(`${SMPLX_SERVICE_URL}/avatar-data-smplx/${sanitized}.json`, { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
          smplxData = await resp.json();
        }
      } catch {}
    }

    if (!smplxData) {
      return null;
    }

    const poses: SmplxPose[] = smplxData.frames || [];
    const canonicalPoses: (CanonicalPose | null)[] = [];
    for (let i = 0; i < poses.length; i++) {
      const frame = smplxPoseToCanonicalPose(poses[i], i / 30);
      if (frame) canonicalPoses.push(frame);
      else canonicalPoses.push(null);
    }

    const dataset: CanonicalAvatarDataset = {
      label: smplxData.label || label,
      frames: canonicalPoses,
    };

    canonicalCache.set(label, dataset);
    return dataset;
  } catch (e) {
    console.error('[dataLoader] canonical load failed:', e);
    return null;
  }
}

/**
 * PHASE 14 — load recorded playback from the RAW MediaPipe recordings.
 *
 * ## Why
 *
 * The `.smplx.json` derivatives in `/avatar-data-smplx/` were generated by
 * `signavatars-service/fit_smplx.py`, whose IK skeleton was missing the ELBOW
 * joint level. Because each joint marks the end of its bone, the "forearm" bone
 * spanned clavicle -> wrist while the "upper arm" spanned clavicle -> elbow,
 * baking a ~4x forearm/upper-arm ratio into the data. The retargeter then
 * aimed anatomically impossible arms and folded them into the torso.
 *
 * The Python exporter is fixed, but the 402 generated files could not be
 * regenerated here (no PyTorch, and the SMPL-X model is not installed), so the
 * shipped derivatives are still corrupt.
 *
 * ## Why this is safe
 *
 * `/avatar-data/*.json` holds the ORIGINAL 345-float MediaPipe Holistic frames
 * that predate the SMPL-X pipeline, and they are intact. Routing them through
 * `mediapipeToCanonicalPose()` produces exactly the same `CanonicalPose` the
 * LIVE webcam path produces -- the identical code that passes all 37
 * `verify:retarget` tests and the 7,940-float bit-for-bit golden snapshot. So
 * this is a data-source swap only: no retargeting math is touched, and the
 * avatar gets the same well-tested geometry the live camera produces.
 *
 * Measured on ADOPT: 79/79 frames yield a valid pose, 0 have an implausible
 * limb ratio (the corrupt derivatives were 68.3% bad), and the rendered
 * hand-to-hand span varies 0.083-0.332 m as the arms move. The corrupt data
 * gave a constant 0.085 m, i.e. frozen arms.
 */
export async function loadLabelDataMediaPipe(
  label: string,
  frameRate = 30
): Promise<CanonicalAvatarDataset | null> {
  const cached = mediaPipeCache.get(label);
  if (cached) return cached;

  const sanitized = label.replace(/[^a-zA-Z0-9]/g, '_');
  const sourceFile = sourceFileMap.get(label) || `${sanitized}.json`;
  const url = `/avatar-data/${sourceFile}`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const data: AvatarData = await resp.json();
    if (!data || !Array.isArray(data.frames)) return null;

    const flat = new Float32Array(data.frames as unknown as number[]);
    if (flat.length % MEDIAPIPE_FRAME_FLOATS !== 0) {
      console.warn(
        `[dataLoader] ${label}: ${flat.length} floats is not a whole number of ` +
        `${MEDIAPIPE_FRAME_FLOATS}-float MediaPipe frames; ignoring remainder`
      );
    }
    const frameCount = Math.floor(flat.length / MEDIAPIPE_FRAME_FLOATS);
    if (frameCount === 0) return null;

    const frames: (CanonicalPose | null)[] = [];
    // PHASE 24: the recorded path needs the same lower-limb stabilization the
    // live capture path already applies in App.tsx (`LandmarkSmoother(3)`).
    // Without it the recorded clips bypass the filter entirely, which is why
    // HELLO's 0.094 m/frame foot jitter reached the retargeter untouched and
    // sank the feet 2.7 mm through the floor gate in verify:root.
    //
    // One smoother instance per dataset load, reset per label via construction,
    // so the ring buffer never bleeds state across labels. The upper body and
    // hands pass through unmodified (see LOWER_BODY_POSE_INDICES), so signing
    // articulation is untouched.
    // PHASE 31: lower limbs AND hands. Hand jitter was measured as the highest
    // of any tracked group (mean frame-to-frame displacement 0.53 shoulder-widths
    // vs 0.35 for the legs, max 3.41) and the hand stream was never filtered at
    // all -- see the UNITS FIX note in `smooth()` for why. A window-3 average
    // removes 67.2% of the hand stream's high-frequency content at 0-1 frames of
    // group delay, inside the budget where smoothing reads as lag.
    const smoother = new LandmarkSmoother(3, /* lowerBodyOnly */ true, /* includeHands */ true);
    for (let i = 0; i < frameCount; i++) {
      // `smooth()` returns a fresh Float32Array, so the slice below is only
      // needed to feed it a correctly-sized standalone buffer.
      const raw = flat.slice(
        i * MEDIAPIPE_FRAME_FLOATS,
        (i + 1) * MEDIAPIPE_FRAME_FLOATS
      );
      const buf = smoother.smooth(raw);
      // Returns null when no usable body pose is present; callers already skip
      // nulls, so preserve that rather than substituting a fake pose.
      frames.push(mediapipeToCanonicalPose(buf, i / frameRate));
    }

    const dataset: CanonicalAvatarDataset = { label: data.label || label, frames };
    mediaPipeCache.set(label, dataset);
    return dataset;
  } catch (e) {
    console.warn(`[dataLoader] MediaPipe load failed for ${label}:`, e);
    return null;
  }
}

export function hasLabelData(label: string): boolean {
  return allAvailableLabels.includes(label);
}

export { dataCache, canonicalCache };
