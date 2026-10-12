/**
 * Diagnostic: quantify how the angled capture rig distorts the
 * 3D components of the DTW tutor's body features.
 *
 * The capture-quality gate measures nose-forward = 0.7-1.1
 * TORSO-lengths corpus-wide (healthy: 0.10-0.15). This script
 * converts that into the units the tutor actually compares —
 * shoulder-widths, after normalizePackedVector — for the three
 * body directions the feature extractor derives from the pose
 * landmarks:
 *
 *   headDir      nose - shoulderMid        (the measured defect)
 *   spineDir     shoulderMid - hipMid      (trunk axis)
 *   shoulderAxis rightShoulder - leftShoulder
 *
 * For each it reports the mean z (depth) component, its
 * frame-to-frame spread, and the fraction of frames whose |z|
 * exceeds 0.1 shoulder-widths (a frontal capture sits near 0).
 * Components that are systematically offset AND stable are rig
 * artifacts (the comparison is then biased); components that
 * merely jitter are noise. Either way, depth is not a comparable
 * signal between the canonical recordings and a live student.
 *
 * Usage: npx tsx scripts/measure-rig-distortion.ts [nFiles]
 */
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { normalizePackedVector } from "../src/features/signSimilarity";

const DATA_DIR = join(process.cwd(), "public", "avatar-data");
const FRAME_FLOATS = 345;
const POSE = 0;
const L_SHOULDER = 11;
const R_SHOULDER = 12;
const L_HIP = 23;
const R_HIP = 24;
const NOSE = 0;

interface Acc {
  sum: number;
  sumSq: number;
  absSum: number;
  overThreshold: number;
  frames: number;
}

function acc(): Acc {
  return { sum: 0, sumSq: 0, absSum: 0, overThreshold: 0, frames: 0 };
}

function add(a: Acc, z: number): void {
  a.sum += z;
  a.sumSq += z * z;
  a.absSum += Math.abs(z);
  if (Math.abs(z) > 0.1) a.overThreshold++;
  a.frames++;
}

function report(name: string, a: Acc): void {
  if (a.frames === 0) {
    console.log(`  ${name}: no frames`);
    return;
  }
  const mean = a.sum / a.frames;
  const meanAbs = a.absSum / a.frames;
  const std = Math.sqrt(Math.max(0, a.sumSq / a.frames - mean * mean));
  const pct = (100 * a.overThreshold) / a.frames;
  console.log(
    `  ${name.padEnd(14)} meanZ=${mean.toFixed(3)}  mean|Z|=${meanAbs.toFixed(3)}` +
      `  frameStd=${std.toFixed(3)}  |z|>0.1: ${pct.toFixed(1)}%`
  );
}

const files = readdirSync(DATA_DIR)
  .filter((f) => f.endsWith(".json") && f !== "manifest.json")
  .sort();
const n = Math.min(files.length, Number(process.argv[2] || 12));
const pick: string[] = [];
for (let i = 0; i < n; i++) {
  pick.push(files[Math.floor((i * files.length) / n)]);
}

const head = acc();
const spine = acc();
const axis = acc();
let noseForwardTorso = acc();

for (const file of pick) {
  const raw = JSON.parse(readFileSync(join(DATA_DIR, file), "utf8")) as {
    frames?: number[];
  };
  const flat = new Float32Array(raw.frames || []);
  const frameCount = Math.floor(flat.length / FRAME_FLOATS);
  const vec = new Float32Array(FRAME_FLOATS);
  for (let i = 0; i < frameCount; i++) {
    const frame = flat.slice(i * FRAME_FLOATS, (i + 1) * FRAME_FLOATS);
    normalizePackedVector(frame, vec);

    const p = (idx: number) => POSE + idx * 3;
    const ls = [vec[p(L_SHOULDER)], vec[p(L_SHOULDER) + 1], vec[p(L_SHOULDER) + 2]];
    const rs = [vec[p(R_SHOULDER)], vec[p(R_SHOULDER) + 1], vec[p(R_SHOULDER) + 2]];
    const lh = [vec[p(L_HIP)], vec[p(L_HIP) + 1], vec[p(L_HIP) + 2]];
    const rh = [vec[p(R_HIP)], vec[p(R_HIP) + 1], vec[p(R_HIP) + 2]];
    const nose = [vec[p(NOSE)], vec[p(NOSE) + 1], vec[p(NOSE) + 2]];

    const mid = (a: number[], b: number[]) =>
      [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
    const sub = (a: number[], b: number[]) =>
      [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const norm = (a: number[]) => {
      const l = Math.hypot(a[0], a[1], a[2]);
      return l < 1e-10 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
    };

    const shoulderMid = mid(ls, rs);
    const hipMid = mid(lh, rh);

    add(head, norm(sub(nose, shoulderMid))[2]);
    add(spine, norm(sub(shoulderMid, hipMid))[2]);
    add(axis, norm(sub(rs, ls))[2]);

    // nose-forward in torso-lengths (the capture-quality metric)
    const torso = Math.hypot(
      shoulderMid[0] - hipMid[0],
      shoulderMid[1] - hipMid[1],
      shoulderMid[2] - hipMid[2]
    );
    if (torso > 1e-4) {
      noseForwardTorso.sum += (shoulderMid[2] - nose[2]) / torso;
      noseForwardTorso.frames++;
    }
  }
}

console.log(`Measured ${pick.length} recordings (${pick.join(", ")}):`);
console.log(`  nose-forward (torso-lengths): mean=${(noseForwardTorso.sum / Math.max(1, noseForwardTorso.frames)).toFixed(3)}  [healthy 0.10-0.15]`);
report("headDir z", head);
report("spineDir z", spine);
report("shoulderAxis z", axis);
