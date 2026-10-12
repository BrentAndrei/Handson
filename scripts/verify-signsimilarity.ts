/**
 * verify:signsimilarity — gate for the DTW practice scorer.
 *
 * Builds synthetic 345-float MediaPipe frames of known geometry
 * (T-pose, arms-up, curled fingers) and asserts the properties the
 * tutor depends on:
 *
 *   1. identical sequences score ~1.0
 *   2. the same sign at a different speed still scores high (DTW
 *      compares shape, not timing)
 *   3. a visibly different pose scores low
 *   4. a one-handed perturbation damages that hand's part score
 *      only (per-part breakdown is real)
 *   5. curl changes are detected (handshape is the dominant signal)
 *   6. too-short attempts return null
 *   7. normalization is idempotent
 *
 * Synthetic frames are deliberately simple: the tests assert
 * relative ordering and thresholds, not absolute scores.
 */
import {
  DEFAULT_SIMILARITY_CONFIG,
  SignSimilarityTracker,
  dtwMeanDistance,
  frameToFeatureVector,
  normalizePackedVector,
  FEATURE_DIM,
} from "../src/features/signSimilarity";

const FRAME_FLOATS = 345;
const POSE_OFFSET = 0;
const LEFT_HAND_OFFSET = 99;
const RIGHT_HAND_OFFSET = 162;
/** Body section of the 49-dim feature vector (signSimilarity.ts). */
const BODY_START = 27;

interface SynthOpts {
  /** 0 = T-pose, 1 = hands overhead. */
  armRaise?: number;
  leftCurl?: number;
  rightCurl?: number;
}

const HAND_SEG = 0.05;

/**
 * Build one synthetic raw MediaPipe frame. Fingers extend outward
 * from the wrist; `curl` rotates each successive phalange by
 * curl*90° toward image-down, so curl=1 gives a 90° bend at every
 * joint (the fist reference the curl remap is calibrated against).
 */
function makeRawFrame(opts: SynthOpts = {}): Float32Array {
  const v = new Float32Array(FRAME_FLOATS);
  const set = (offset: number, index: number, x: number, y: number, z: number) => {
    const b = offset + index * 3;
    v[b] = x;
    v[b + 1] = y;
    v[b + 2] = z;
  };
  const raise = opts.armRaise ?? 0;

  set(POSE_OFFSET, 0, 0, 0.1, 0); // nose
  set(POSE_OFFSET, 11, -0.15, 0, 0); // L shoulder
  set(POSE_OFFSET, 12, 0.15, 0, 0); // R shoulder
  set(POSE_OFFSET, 13, -0.35, 0.05 - 0.6 * raise, 0); // L elbow
  set(POSE_OFFSET, 14, 0.35, 0.05 - 0.6 * raise, 0); // R elbow
  set(POSE_OFFSET, 15, -0.55, 0.05 - 0.75 * raise, 0); // L wrist
  set(POSE_OFFSET, 16, 0.55, 0.05 - 0.75 * raise, 0); // R wrist
  set(POSE_OFFSET, 23, -0.12, 0.35, 0); // L hip
  set(POSE_OFFSET, 24, 0.12, 0.35, 0); // R hip

  // Finger MCP indices: thumb 1, index 5, middle 9, ring 13, pinky 17.
  // Each finger gets a slight y spread so the palm normal is well defined.
  const fingerDy = [-0.02, -0.03, 0, 0.03, 0.06];
  const buildHand = (
    handOffset: number,
    wristX: number,
    wristY: number,
    curl: number,
    outward: number
  ) => {
    set(handOffset, 0, wristX, wristY, 0);
    for (let f = 0; f < 5; f++) {
      const mcp = 1 + f * 4;
      const dy = fingerDy[f];
      for (let k = 1; k <= 3; k++) {
        // Position is the sum of segments 1..k; segment s
        // rotates curl*90°*s from the outward axis toward
        // image-down (+y). curl=0 keeps all segments collinear.
        let x = wristX;
        let y = wristY + dy;
        for (let s = 1; s <= k; s++) {
          const a = curl * (Math.PI / 2) * s;
          x += Math.cos(a) * outward * HAND_SEG;
          y += Math.sin(a) * HAND_SEG;
        }
        set(handOffset, mcp + k - 1, x, y, 0);
      }
      // Tip (4th landmark of the finger): one more small segment.
      let x = wristX;
      let y = wristY + dy;
      for (let s = 1; s <= 3; s++) {
        const a = curl * (Math.PI / 2) * s;
        x += Math.cos(a) * outward * HAND_SEG;
        y += Math.sin(a) * HAND_SEG;
      }
      const aTip = curl * (Math.PI / 2) * 4;
      x += Math.cos(aTip) * outward * HAND_SEG * 0.6;
      y += Math.sin(aTip) * HAND_SEG * 0.6;
      set(handOffset, mcp + 3, x, y, 0);
    }
  };

  const wristY = 0.05 - 0.75 * raise;
  buildHand(LEFT_HAND_OFFSET, -0.55, wristY, opts.leftCurl ?? 0, -1);
  buildHand(RIGHT_HAND_OFFSET, 0.55, wristY, opts.rightCurl ?? 0, 1);
  return v;
}

function makeSequence(
  count: number,
  opts: SynthOpts = {}
): Float32Array[] {
  const frames: Float32Array[] = [];
  for (let i = 0; i < count; i++) frames.push(makeRawFrame(opts));
  return frames;
}

let checks = 0;
let failures = 0;

function check(name: string, passed: boolean, detail: string): void {
  checks++;
  if (!passed) failures++;
  console.log(`  ${passed ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
}

function scoreOf(live: Float32Array[], canonical: Float32Array[], label = "SYN"): number {
  const tracker = new SignSimilarityTracker(label, canonical);
  for (const f of live) {
    const norm = new Float32Array(FRAME_FLOATS);
    normalizePackedVector(f, norm);
    tracker.pushLiveFrame(norm);
  }
  const result = tracker.finish();
  if (!result) return -1;
  return result.score;
}

// ---------------------------------------------------------------------------
console.log("S1. Identical sequences score ~1.0");
{
  const seq = makeSequence(40);
  const s = scoreOf(seq, seq);
  check("identical score >= 0.99", s >= 0.99, `score=${s.toFixed(4)}`);
}

console.log("S2. Same sign, different speed (2x stretched) still matches");
{
  const base = makeSequence(40);
  const stretched: Float32Array[] = [];
  for (const f of base) {
    stretched.push(f);
    stretched.push(f); // duplicate every frame -> 2x duration
  }
  const s = scoreOf(stretched, base);
  check("stretched score >= 0.9", s >= 0.9, `score=${s.toFixed(4)}`);
}

console.log("S3. Different pose (arms overhead) scores low");
{
  const base = makeSequence(40);
  const armsUp = makeSequence(40, { armRaise: 1 });
  const sSame = scoreOf(base, base);
  const sDiff = scoreOf(armsUp, base);
  check("arms-up score <= 0.6", sDiff <= 0.6, `score=${sDiff.toFixed(4)}`);
  check("arms-up < identical", sDiff < sSame, `${sDiff.toFixed(4)} < ${sSame.toFixed(4)}`);
}

console.log("S4. One-hand perturbation damages only that hand's part score");
{
  const base = makeSequence(40);
  const perturbed = makeSequence(40, { leftCurl: 1 });
  const tracker = new SignSimilarityTracker("SYN", base);
  for (const f of perturbed) {
    const norm = new Float32Array(FRAME_FLOATS);
    normalizePackedVector(f, norm);
    tracker.pushLiveFrame(norm);
  }
  const r = tracker.finish();
  if (!r) {
    check("result not null", false, "finish() returned null");
  } else {
    check(
      "leftHand < rightHand",
      r.leftHand < r.rightHand,
      `L=${r.leftHand.toFixed(4)} R=${r.rightHand.toFixed(4)}`
    );
    check(
      "leftHand < body",
      r.leftHand < r.body,
      `L=${r.leftHand.toFixed(4)} body=${r.body.toFixed(4)}`
    );
    check(
      "rightHand ~ body (untouched parts stay high)",
      r.rightHand > 0.9 && r.body > 0.9,
      `R=${r.rightHand.toFixed(4)} body=${r.body.toFixed(4)}`
    );
  }
}

console.log("S5. Curl changes are detected on both hands");
{
  const base = makeSequence(40);
  const fists = makeSequence(40, { leftCurl: 1, rightCurl: 1 });
  const tracker = new SignSimilarityTracker("SYN", base);
  for (const f of fists) {
    const norm = new Float32Array(FRAME_FLOATS);
    normalizePackedVector(f, norm);
    tracker.pushLiveFrame(norm);
  }
  const r = tracker.finish();
  if (!r) {
    check("result not null", false, "finish() returned null");
  } else {
    check(
      "both hand scores < body",
      r.leftHand < r.body && r.rightHand < r.body,
      `L=${r.leftHand.toFixed(4)} R=${r.rightHand.toFixed(4)} body=${r.body.toFixed(4)}`
    );
    check(
      "overall drops below identical",
      r.score < 0.9,
      `score=${r.score.toFixed(4)}`
    );
  }
}

console.log("S6. Too-short attempts return null");
{
  const base = makeSequence(40);
  const short = makeSequence(DEFAULT_SIMILARITY_CONFIG.minLiveFrames - 1);
  const tracker = new SignSimilarityTracker("SYN", base);
  for (const f of short) {
    const norm = new Float32Array(FRAME_FLOATS);
    normalizePackedVector(f, norm);
    tracker.pushLiveFrame(norm);
  }
  check(
    "short attempt -> null",
    tracker.finish() === null,
    `minLiveFrames=${DEFAULT_SIMILARITY_CONFIG.minLiveFrames}`
  );
}

console.log("S7. Normalization is idempotent");
{
  const raw = makeRawFrame({ armRaise: 0.5, leftCurl: 0.5 });
  const once = new Float32Array(FRAME_FLOATS);
  normalizePackedVector(raw, once);
  const twice = new Float32Array(FRAME_FLOATS);
  normalizePackedVector(once, twice);
  let maxDelta = 0;
  for (let i = 0; i < FRAME_FLOATS; i++) {
    const d = Math.abs(once[i] - twice[i]);
    if (d > maxDelta) maxDelta = d;
  }
  check("idempotent (max delta < 1e-6)", maxDelta < 1e-6, `maxDelta=${maxDelta.toExponential(2)}`);
}

console.log("S8. Feature vector is finite and correctly sized");
{
  const raw = makeRawFrame();
  const norm = new Float32Array(FRAME_FLOATS);
  normalizePackedVector(raw, norm);
  const feats = frameToFeatureVector(norm);
  check(
    "feature dim = 49",
    feats.length === FEATURE_DIM,
    `length=${feats.length}`
  );
  check(
    "all finite",
    Array.from(feats).every(Number.isFinite),
    ""
  );
  // Curls of an open synthetic hand must be ~0 (collinear segments).
  const openCurls = feats.slice(BODY_START, BODY_START + 11).slice(6);
  check(
    "open-hand curls ~ 0",
    openCurls.every((c) => c < 0.05),
    `curls=[${Array.from(openCurls).map((c) => c.toFixed(3)).join(",")}]`
  );
}

console.log("S9. DTW distance is symmetric and bounded");
{
  const a = makeSequence(30, { armRaise: 0.3 });
  const b = makeSequence(30, { armRaise: 0.6 });
  const fa = a.map((f) => {
    const n = new Float32Array(FRAME_FLOATS);
    normalizePackedVector(f, n);
    return frameToFeatureVector(n, new Float32Array(FEATURE_DIM));
  });
  const fb = b.map((f) => {
    const n = new Float32Array(FRAME_FLOATS);
    normalizePackedVector(f, n);
    return frameToFeatureVector(n, new Float32Array(FEATURE_DIM));
  });
  const dAB = dtwMeanDistance(fa, fb);
  const dBA = dtwMeanDistance(fb, fa);
  check(
    "symmetric (|dAB - dBA| < 1e-9)",
    Math.abs(dAB - dBA) < 1e-9,
    `dAB=${dAB.toFixed(6)} dBA=${dBA.toFixed(6)}`
  );
  check("finite", Number.isFinite(dAB), `dAB=${dAB}`);
}

console.log("S10. Rig depth distortion is rejected (frontal-plane body)");
{
  // The shipped corpus was captured with an angled rig: the
  // nose sits ~0.9 torso-lengths forward of the shoulder line
  // (healthy 0.10-0.15) and the trunk axis carries a stable
  // depth offset (measured: headDir z = -0.925 shoulder-widths
  // on 100% of frames — scripts/measure-rig-distortion.ts).
  // A student whose ONLY difference from the reference is that
  // rig artifact must still score as a perfect match.
  //
  // Nose ~3 shoulder-widths and hips ~1 shoulder-width forward
  // in raw units (this fixture's shoulder width is 0.3, so the
  // normalized offsets are ~10x and ~3.3x the measured defect).
  const base = makeSequence(40);
  const distorted = makeSequence(40);
  for (const f of distorted) {
    f[POSE_OFFSET + 0 * 3 + 2] = -0.9; // nose z
    f[POSE_OFFSET + 23 * 3 + 2] = -0.3; // L hip z
    f[POSE_OFFSET + 24 * 3 + 2] = -0.3; // R hip z
  }
  const s = scoreOf(distorted, base);
  check(
    "rig-depth-only difference scores >= 0.99",
    s >= 0.99,
    `score=${s.toFixed(4)} (3D headDir/spineDir would score ~0.1)`
  );
}

console.log("");
if (failures === 0) {
  console.log(`VERIFY PASSED: ${checks} checks, 0 failures`);
} else {
  console.log(`VERIFY FAILED: ${failures}/${checks} checks failed`);
  process.exit(1);
}
