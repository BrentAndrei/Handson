import { retargetFrames, smoothFramesSpline, type AvatarFrame, type ParentWorldQuats } from './retarget';

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

export function hasLabelData(label: string): boolean {
  return allAvailableLabels.includes(label);
}

export { dataCache };
