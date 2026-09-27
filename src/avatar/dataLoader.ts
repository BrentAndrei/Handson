import { retargetFrames, smoothFramesSpline, type AvatarFrame, type ParentWorldQuats } from './retarget';
import { smplxToAvatarFrames, type SmplxFrame } from './smplxConverter';

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

const dataCache: Map<string, AvatarDataset> = new Map();
let allAvailableLabels: string[] = [];
let labelsLoaded = false;

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
    allAvailableLabels = labels;
    labelsLoaded = true;
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
  const url = `/avatar-data/${sanitized}.json`;

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
    try {
      const resp = await fetch(smplxUrl, { signal: AbortSignal.timeout(3000) });
      if (resp.ok) {
        smplxData = await resp.json();
      }
    } catch {
      // Service not reachable, fall back to live fitting
    }

    if (!smplxData) {
      // Fall back to current quaternion retargeting
      const url = `/avatar-data/${sanitized}.json`;
      const resp = await fetch(url);
      if (!resp.ok) return null;
      const data: AvatarData = await resp.json();

      const landmarksData = data.frames.flat();
      const fitResp = await fetch(`${SMPLX_SERVICE_URL}/fit-sequence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          frames: [landmarksData],
          parent_world_quats: parentWorldQuats,
        }),
      });
      if (!fitResp.ok) return null;
      const fitResult = await fitResp.json();
      smplxData = { label, frames: fitResult.frames || [], frameRate: 30 };
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

export function hasLabelData(label: string): boolean {
  return allAvailableLabels.includes(label);
}

export { dataCache };
