/**
 * Frame quality gate verification — drives the gate with REAL
 * packed MediaPipe frames from the corpus.
 *
 * The gate's job:
 *   1. Frames with no usable hands are rejected (usable=false)
 *   2. Consecutive hand-absent frames flip isRest=true
 *   3. A real sign's frames are all accepted (no false rejects)
 *   4. Motion is measured in shoulder-width units
 *
 * A regression in either direction breaks conversational
 * accuracy: too permissive and garbage leaks into the window;
 * too strict and real signs are dropped.
 *
 * Run: npx tsx scripts/verify-frame-quality.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import {
  FrameQualityGate,
  handGroupSpread,
  MIN_HAND_SPREAD,
  REST_FRAME_THRESHOLD,
} from "../src/inference/frameQuality";
import { selectFeatures, FEATURE_DIM } from "../src/features/featureSelect";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const RAW_FLOATS = 345;

function loadFrames(file: string): Float32Array {
  const j = JSON.parse(fs.readFileSync(file, "utf8")) as { frames?: number[] };
  return new Float32Array(j.frames ?? []);
}

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`  [${ok ? "ok" : "FAIL"}] ${name}${detail ? ` — ${detail}` : ""}`);
}

function main() {
  console.log("=== FRAME QUALITY GATE VERIFY ===\n");

  // ---- 1. FEATURE_OFFSET sanity: hand spreads computed on the
  // feature vector must match those computed on the raw frame.
  console.log("-- feature-vector offsets --");
  const raw = loadFrames(path.join(DATA_DIR, "FOUR.json"));
  const frameCount = Math.floor(raw.length / RAW_FLOATS);
  // Find a frame with a left hand present in the raw buffer.
  let found = -1;
  for (let fi = 0; fi < frameCount; fi++) {
    const frame = raw.slice(fi * RAW_FLOATS, (fi + 1) * RAW_FLOATS);
    const leftSpread = handGroupSpread(frame, 99);
    const rightSpread = handGroupSpread(frame, 162);
    if (leftSpread >= MIN_HAND_SPREAD || rightSpread >= MIN_HAND_SPREAD) {
      found = fi;
      break;
    }
  }
  check("found a frame with a hand", found >= 0, `frame ${found}`);
  if (found >= 0) {
    const frame = raw.slice(found * RAW_FLOATS, (found + 1) * RAW_FLOATS);
    const features = selectFeatures(frame);
    check("feature dim is 345", FEATURE_DIM === 345, `FEATURE_DIM=${FEATURE_DIM}`);
    // Raw left/right hand spread must equal feature-vector spread.
    const rawLeft = handGroupSpread(frame, 99);
    const rawRight = handGroupSpread(frame, 162);
    const featLeft = handGroupSpread(features, 99);
    const featRight = handGroupSpread(features, 162);
    check(
      "left hand spread matches on feature vector",
      Math.abs(rawLeft - featLeft) < 1e-5,
      `raw=${rawLeft.toFixed(4)} feat=${featLeft.toFixed(4)}`
    );
    check(
      "right hand spread matches on feature vector",
      Math.abs(rawRight - featRight) < 1e-5,
      `raw=${rawRight.toFixed(4)} feat=${featRight.toFixed(4)}`
    );
  }

  // ---- 2. Real sign frames are all accepted (no false rejects).
  console.log("\n-- real-sign acceptance --");
  for (const rec of ["FOUR", "FIVE", "THANK_YOU", "KNOW"]) {
    const file = path.join(DATA_DIR, `${rec}.json`);
    if (!fs.existsSync(file)) continue;
    const data = loadFrames(file);
    const n = Math.floor(data.length / RAW_FLOATS);

    const gate = new FrameQualityGate();
    let accepted = 0;
    let restFrames = 0;
    let handFrames = 0;
    let firstHand = -1;
    for (let fi = 0; fi < n; fi++) {
      const features = selectFeatures(data.slice(fi * RAW_FLOATS, (fi + 1) * RAW_FLOATS));
      const q = gate.evaluate(features);
      if (q.usable) {
        accepted++;
        if (firstHand < 0) firstHand = fi;
      }
      if (q.isRest) restFrames++;
      const ls = handGroupSpread(features, 99);
      const rs = handGroupSpread(features, 162);
      if (ls >= MIN_HAND_SPREAD || rs >= MIN_HAND_SPREAD) handFrames++;
    }
    check(
      `${rec}: every hand-bearing frame accepted`,
      accepted === handFrames,
      `accepted=${accepted} handFrames=${handFrames} total=${n}`
    );
    check(
      `${rec}: rest detected when hands absent`,
      restFrames > 0,
      `restFrames=${restFrames}/${n}`
    );
  }

  // ---- 3. Rest-state reset: a hand-absent run must flip
  // isRest, and a returning hand must clear it.
  console.log("\n-- rest hysteresis --");
  {
    const gate = new FrameQualityGate();
    const file = path.join(DATA_DIR, "FOUR.json");
    const data = loadFrames(file);
    const n = Math.floor(data.length / RAW_FLOATS);
    // Absent-hand frame: take one and zero BOTH hand groups.
    const blank = selectFeatures(data.slice(0, RAW_FLOATS));
    for (let i = 0; i < 126; i++) {
      blank[99 + i] = 0;
      blank[162 + i] = 0;
    }
    // Real hand frame.
    let handFrame: Float32Array | null = null;
    for (let fi = 0; fi < n; fi++) {
      const features = selectFeatures(data.slice(fi * RAW_FLOATS, (fi + 1) * RAW_FLOATS));
      if (handGroupSpread(features, 99) >= MIN_HAND_SPREAD) { handFrame = features; break; }
    }
    check("have a real hand frame for hysteresis test", handFrame !== null);
    if (handFrame) {
      let flippedAt = -1;
      for (let fi = 0; fi < 10; fi++) {
        const q = gate.evaluate(blank);
        if (q.isRest && flippedAt < 0) flippedAt = fi + 1;
        check(`blank frame ${fi} rejected`, !q.usable);
      }
      check(
        `isRest flips after ${REST_FRAME_THRESHOLD} blank frames`,
        flippedAt === REST_FRAME_THRESHOLD,
        `flippedAt=${flippedAt}`
      );
      const back = gate.evaluate(handFrame);
      check("returning hand clears rest", !back.isRest && back.usable);
    }
  }

  // ---- 4. Motion is measured and jitter is flagged.
  console.log("\n-- motion / jitter --");
  {
    const gate = new FrameQualityGate();
    const data = loadFrames(path.join(DATA_DIR, "FOUR.json"));
    const n = Math.floor(data.length / RAW_FLOATS);
    let maxMotion = 0;
    let jitterCount = 0;
    let evaluated = 0;
    for (let fi = 0; fi < n; fi++) {
      const features = selectFeatures(data.slice(fi * RAW_FLOATS, (fi + 1) * RAW_FLOATS));
      const q = gate.evaluate(features);
      if (evaluated > 0) {
        maxMotion = Math.max(maxMotion, q.motion);
        if (q.issues.some((i) => i.startsWith("jump"))) jitterCount++;
      }
      evaluated++;
    }
    check(
      "motion stays finite and bounded",
      maxMotion >= 0 && Number.isFinite(maxMotion),
      `maxMotion=${maxMotion.toFixed(3)} sh-w`
    );
    console.log(`  (info) jitter-flagged frames: ${jitterCount}/${evaluated}`);
  }

  console.log(`\n${failures === 0 ? "VERIFY PASSED" : `VERIFY FAILED: ${failures} checks`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
