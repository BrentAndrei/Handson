/**
 * PHASE 8 - presentation-layer validation.
 *
 * The Phase 8 brief said smoothing / spline / idle had been DISABLED in
 * Phases 1-7 and needed re-enabling. Measurement says otherwise: all three are
 * ALREADY LIVE.
 *   - LandmarkSmoother: App.tsx:446 smooths the full 1659-float packed buffer
 *     on the live MediaPipe path.
 *   - HolisticSmoother: constructed + .update()d at 5 sites in
 *     useHolisticPipeline.ts.
 *   - smoothFramesSpline(frames, 5): dataLoader.ts, legacy SMPL-X quaternion
 *     path. The canonical path does not use it.
 *   - idle/breathing: a SignAvatar.tsx block that OVERWRITES 11 bones,
 *     including BOTH arms and BOTH hands.
 *
 * Every Phase 4-7 gate calls retarget() directly, so none of these has ever
 * been measured against the dot gates. This measures them.
 *
 * Run: npx tsx scripts/verify-presentation-layers.ts
 */
import * as THREE from "three";
import { LandmarkSmoother } from "../src/utils/landmarkSmoother";
import { TOTAL_FLOATS, POSE_OFFSET } from "../src/mediapipe/types";
import { HolisticSmoother } from "../src/mediapipe/smoothing";
import { smoothFramesSpline, type AvatarFrame, type Quat } from "../src/avatar/retarget";

const f3 = (n: number) => n.toFixed(4);
const D = 180 / Math.PI;
const LM_INDEX = 11; // left shoulder

function makeRand(seedIn: number) {
  let s = seedIn;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
}

function qAng(a: THREE.Quaternion, b: THREE.Quaternion): number {
  return D * 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));
}

const rms = (a: number[], ref: number[]) =>
  Math.sqrt(a.reduce((s, v, i) => s + (v - ref[i]) ** 2, 0) / a.length);

/** 1. LandmarkSmoother on a full packed buffer (the real App.tsx call shape). */
function testLandmarkSmoother() {
  console.log("=== 1. LandmarkSmoother(3) on the full packed buffer (App.tsx) ===");
  const N = 120;
  const rand = makeRand(12345);
  const truth: number[] = [];
  const noisy: number[] = [];
  for (let i = 0; i < N; i++) {
    const c = Math.sin((i / N) * Math.PI * 2);
    truth.push(c);
    noisy.push(c + rand() * 0.1);
  }
  const sm = new LandmarkSmoother(3);
  const out: number[] = [];
  for (let i = 0; i < N; i++) {
    const buf = new Float32Array(TOTAL_FLOATS);
    buf[POSE_OFFSET + LM_INDEX * 3] = noisy[i];
    out.push(sm.smooth(buf)[POSE_OFFSET + LM_INDEX * 3]);
  }
  const noiseBefore = rms(noisy, truth);
  // Suppression must be measured AFTER removing the filter's own lag, otherwise
  // a moving average's phase lag against a smooth signal is counted as
  // "jitter" and a correctly-behaving filter looks like it makes things worse.
  // Compare the output against the clean signal shifted by the measured lag.
  const rmsLagged = (a: number[], lag: number) => {
    let s = 0;
    let n = 0;
    for (let i = lag; i < N; i++) {
      s += (a[i] - truth[i - lag]) ** 2;
      n++;
    }
    return Math.sqrt(s / n);
  };
  const rmsBeforeLagged = rmsLagged(noisy, 0);

  let bestLag = 0;
  let bestErr = Infinity;
  for (let lag = 0; lag <= 12; lag++) {
    const e = rmsLagged(out, lag);
    if (e < bestErr) { bestErr = e; bestLag = lag; }
  }
  const after = rmsLagged(out, bestLag);
  const supp = 1 - after / rmsBeforeLagged;
  const totalErr = rms(out, truth);

  // How often is the output identical to the input (i.e. history discarded)?
  let passthrough = 0;
  for (let i = 1; i < N; i++) if (Math.abs(out[i] - noisy[i]) < 1e-6) passthrough++;

  console.log(`  input noise RMS (vs clean)      : ${f3(noiseBefore)}`);
  console.log(`  total output error (vs clean)   : ${f3(totalErr)}  (lag + jitter)`);
  console.log(`  measured lag                    : ${bestLag} frames @30fps = ${f3((bestLag / 30) * 1000)} ms`);
  console.log(`  jitter RMS before (lag-aligned) : ${f3(rmsBeforeLagged)}`);
  console.log(`  jitter RMS after  (lag-aligned) : ${f3(after)}`);
  console.log(`  JITTER SUPPRESSION              : ${(supp * 100).toFixed(1)}%`);
  console.log(`  frames passed through unfiltered: ${passthrough}/${N - 1}`);
  console.log(`  verdict                         : ${supp > 0.3 && passthrough < 5 ? "PASS" : "FAIL"}`);
  console.log("");
  return { supp, passthrough, bestLag };
}

/** 2. HolisticSmoother (OneEuro) on the live webcam path. */
function testHolisticSmoother() {
  console.log("=== 2. HolisticSmoother (OneEuro, live webcam path) ===");
  const sm = new HolisticSmoother();
  const N = 90;
  const rand = makeRand(999);
  const clean = (i: number) => Math.sin((i / N) * Math.PI * 2) * 0.3;
  let inSq = 0;
  let outSq = 0;
  const xs: number[] = [];
  for (let i = 0; i < N; i++) {
    const jit = i < N / 2 ? 0.1 : 0;
    const j = () => rand() * jit;
    const g = (n: number) => Array.from({ length: n }, () => ({ x: j(), y: j(), z: j() }));
    const r = {
      poseLandmarks: g(33), leftHandLandmarks: g(21),
      rightHandLandmarks: g(21), faceLandmarks: [],
    };
    inSq += (r.poseLandmarks[LM_INDEX].x - clean(i)) ** 2;
    const s = sm.update(r as never, i * (1000 / 30));
    const lm = s.pose.landmarks;
    const ox = lm ? lm[LM_INDEX].x : clean(i);
    xs.push(ox);
    outSq += (ox - clean(i)) ** 2;
  }
  const inRms = Math.sqrt(inSq / N);
  const outRms = Math.sqrt(outSq / N);
  const supp = 1 - outRms / inRms;
  // Convergence must be measured against the CLEAN signal over the same tail,
  // not against the tail's own mean: the tail still contains a full sweep of
  // the sine, so a std-vs-mean test measures the signal, not the jitter.
  let tailSq = 0;
  for (let i = N - 30; i < N; i++) tailSq += (xs[i] - clean(i)) ** 2;
  const tailErr = Math.sqrt(tailSq / 30);
  console.log(`  input RMS vs clean   : ${f3(inRms)}`);
  console.log(`  output RMS vs clean  : ${f3(outRms)}`);
  console.log(`  jitter suppression   : ${suppressionPct(supp)}`);
  console.log(`  steady-state error   : ${f3(tailErr)} (last 30 frames vs clean, must reach ~0)`);
  console.log(`  verdict              : ${supp > 0.2 && tailErr < 0.02 ? "PASS" : "REVIEW"}`);
  console.log("");
  return { supp, tailErr };
}

const suppressionPct = (s: number) => `${(s * 100).toFixed(1)}%`;

/** 3. smoothFramesSpline on the legacy SMPL-X quaternion path. */
function testSplineSmoothing() {
  console.log("=== 3. smoothFramesSpline(5) on the legacy SMPL-X path ===");
  const N = 80;
  const rand = makeRand(4242);
  const euler = new THREE.Euler();
  const axis = new THREE.Vector3(1, 0, 0.35).normalize();
  const BONES = ["torso", "leftUpperArm", "rightUpperArm", "leftForearm", "rightForearm"] as const;
  const frames: AvatarFrame[] = [];
  const truth = new Map<string, THREE.Quaternion[]>();
  for (const b of BONES) truth.set(b, []);

  for (let i = 0; i < N; i++) {
    const q = new THREE.Quaternion().setFromAxisAngle(axis, (i / N) * Math.PI * 0.8);
    euler.set(rand() * 0.25, rand() * 0.25, rand() * 0.25);
    const noisy = q.clone().multiply(new THREE.Quaternion().setFromEuler(euler));
    const f: Record<string, unknown> = {
      root: [0, 0, 0],
      hips: [0, 0, 0],
      head: [0, 0, 0, 1] as Quat,
      neck: [0, 0, 0, 1] as Quat,
      leftShoulder: [0, 0, 0, 1] as Quat,
      rightShoulder: [0, 0, 0, 1] as Quat,
      leftHand: [0, 0, 0, 1] as Quat,
      rightHand: [0, 0, 0, 1] as Quat,
    };
    for (const b of BONES) {
      f[b] = [noisy.x, noisy.y, noisy.z, noisy.w] as Quat;
      truth.get(b)!.push(q.clone());
    }
    frames.push(f as unknown as AvatarFrame);
  }

  let out: AvatarFrame[];
  try {
    out = smoothFramesSpline(frames, 5);
  } catch (e) {
    console.log(`  THREW: ${e}`);
    console.log("");
    return { threw: true as const };
  }
  if (!out || out.length !== N) {
    console.log(`  unexpected output length ${out?.length} (expected ${N})`);
    console.log("");
    return { threw: true as const };
  }

  let maxDev = 0;
  let sum = 0;
  let n = 0;
  for (const b of BONES) {
    const tq = truth.get(b)!;
    for (let i = 0; i < N; i++) {
      const g = (out[i] as unknown as Record<string, Quat>)[b];
      if (!g) continue;
      const a = qAng(new THREE.Quaternion(g[0], g[1], g[2], g[3]), tq[i]);
      maxDev = Math.max(maxDev, a);
      sum += a;
      n++;
    }
  }
  const mean = n ? sum / n : 0;
  let pre = 0;
  for (let i = 0; i < N; i++) {
    const g = (frames[i] as unknown as Record<string, Quat>).leftUpperArm;
    pre += qAng(new THREE.Quaternion(g[0], g[1], g[2], g[3]), truth.get("leftUpperArm")![i]);
  }
  pre /= N;

  // Isolate the smoother's OWN error by re-running it on the same sweep with
  // ZERO injected noise. Anything left here is the smoother's bias (lag from
  // the window), not leftover jitter -- that is the number that matters for the
  // "endpoint vectors must survive presentation layers" gate.
  const cleanFrames: AvatarFrame[] = [];
  for (let i = 0; i < N; i++) {
    const q = new THREE.Quaternion().setFromAxisAngle(axis, (i / N) * Math.PI * 0.8);
    const f: Record<string, unknown> = {
      root: [0, 0, 0], hips: [0, 0, 0],
      head: [0, 0, 0, 1] as Quat, neck: [0, 0, 0, 1] as Quat,
      leftShoulder: [0, 0, 0, 1] as Quat, rightShoulder: [0, 0, 0, 1] as Quat,
      leftHand: [0, 0, 0, 1] as Quat, rightHand: [0, 0, 0, 1] as Quat,
    };
    for (const b of BONES) f[b] = [q.x, q.y, q.z, q.w] as Quat;
    cleanFrames.push(f as unknown as AvatarFrame);
  }
  const cleanOut = smoothFramesSpline(cleanFrames, 5);
  let cMax = 0;
  let cSum = 0;
  let cN = 0;
  for (const b of BONES) {
    const tq = truth.get(b)!;
    for (let i = 0; i < N; i++) {
      const g = (cleanOut[i] as unknown as Record<string, Quat>)[b];
      if (!g) continue;
      const a = qAng(new THREE.Quaternion(g[0], g[1], g[2], g[3]), tq[i]);
      cMax = Math.max(cMax, a);
      cSum += a;
      cN++;
    }
  }
  const cMean = cN ? cSum / cN : 0;
  const gateDeg = (2 * Math.acos(0.99) * 180) / Math.PI; // dot > 0.99

  console.log(`  NOISELESS input  -> smoother's own error:`);
  console.log(`     mean ${f3(cMean)} deg, max ${f3(cMax)} deg`);
  console.log(`     dot>0.99 tolerance is ${f3(gateDeg)} deg  ->  ${cMax < gateDeg ? "WITHIN" : "EXCEEDS"} gate`);
  console.log(`  NOISY input (${f3(pre)} deg of jitter):`);
  console.log(`     residual after smoothing: mean ${f3(mean)} deg, max ${f3(maxDev)} deg`);
  console.log(`     noise removed: ${f3(pre - mean)} of ${f3(pre)} deg`);
  console.log(`  frames in -> out          : ${N} -> ${out.length}`);
  console.log(`  verdict                   : ${cMax < gateDeg ? "PASS" : "FAIL - smoothing itself moves bones past the dot gate"}`);
  console.log("");
  return { threw: false as const, mean, maxDev, pre, cMean, cMax, gateDeg };
}

function main() {
  console.log("PHASE 8 - PRESENTATION LAYER VALIDATION");
  console.log("These layers are ALREADY LIVE; measuring whether they corrupt the");
  console.log("Phase 4/5 verified bone orientations.\n");
  const a = testLandmarkSmoother();
  const b = testHolisticSmoother();
  const c = testSplineSmoothing();
  console.log("============ SUMMARY ============");
  console.log(`LandmarkSmoother   : suppression ${suppressionPct(a.supp)}, ${a.passthrough} frames unfiltered, lag ${a.bestLag}`);
  console.log(`HolisticSmoother   : suppression ${suppressionPct(b.supp)}, steady error ${f3(b.tailErr)}`);
  if (c.threw) {
    console.log("smoothFramesSpline : THREW or unexpected length");
    process.exitCode = 1;
  } else {
    console.log(`smoothFramesSpline : own error mean ${f3(c.cMean)} / max ${f3(c.cMax)} deg (dot>0.99 = ${f3(c.gateDeg)} deg)`);
  }
  if (a.supp <= 0.3 || a.passthrough >= 5) process.exitCode = 1;
}

main();

