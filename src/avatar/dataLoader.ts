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

export function hasLabelData(label: string): boolean {
  return allAvailableLabels.includes(label);
}

export { dataCache };
