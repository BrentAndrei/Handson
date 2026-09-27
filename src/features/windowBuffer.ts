export const WINDOW_SIZE = 16;
// Requested range is 4-8; 6 sits in the middle — dense enough to catch
// short signs without emitting (and running inference) every single frame.
export const DEFAULT_STRIDE = 6;

/**
 * Fixed-size ring buffer of feature frames for a LIVE sliding window.
 * Push a new frame every capture tick; once WINDOW_SIZE frames have been
 * pushed, isReady() is true and getWindow() returns the most recent
 * WINDOW_SIZE frames flattened to [WINDOW_SIZE * featureDim] (row-major:
 * frame 0's features, then frame 1's, ...) — the layout both the training
 * script and the tfjs inference code expect.
 *
 * `stride` controls shouldEmit(): once the window has filled, it only
 * returns true every `stride` frames, so live inference isn't re-running
 * the model on every single frame once warmed up.
 */
export class SlidingWindowBuffer {
  private frames: Float32Array[] = [];
  private framesSinceEmit = 0;

  constructor(
    private readonly featureDim: number,
    private readonly windowSize: number = WINDOW_SIZE,
    private readonly stride: number = DEFAULT_STRIDE
  ) {}

  push(frame: Float32Array): void {
    this.frames.push(frame);
    if (this.frames.length > this.windowSize) this.frames.shift();
    this.framesSinceEmit++;
  }

  isReady(): boolean {
    return this.frames.length === this.windowSize;
  }

  /** True once every `stride` frames after the buffer first fills. Resets
   *  the internal counter whenever it returns true. */
  shouldEmit(): boolean {
    if (!this.isReady()) return false;
    if (this.framesSinceEmit < this.stride) return false;
    this.framesSinceEmit = 0;
    return true;
  }

  getWindow(): Float32Array {
    const out = new Float32Array(this.windowSize * this.featureDim);
    for (let i = 0; i < this.frames.length; i++) {
      out.set(this.frames[i], i * this.featureDim);
    }
    return out;
  }

  reset(): void {
    this.frames = [];
    this.framesSinceEmit = 0;
  }
}

/**
 * Batch-slices a full recorded clip (one already-feature-selected
 * Float32Array per frame) into overlapping windowSize-length windows at
 * the given stride. Used by the dataset recorder: record a clip
 * continuously while the user performs one sign a few times, then slice it
 * into many labeled training windows instead of requiring perfectly
 * trimmed single repetitions.
 */
export function sliceIntoWindows(
  clip: Float32Array[],
  featureDim: number,
  windowSize: number = WINDOW_SIZE,
  stride: number = DEFAULT_STRIDE
): Float32Array[] {
  const windows: Float32Array[] = [];
  if (clip.length < windowSize) return windows;
  for (let start = 0; start + windowSize <= clip.length; start += stride) {
    const out = new Float32Array(windowSize * featureDim);
    for (let i = 0; i < windowSize; i++) {
      out.set(clip[start + i], i * featureDim);
    }
    windows.push(out);
  }
  return windows;
}
