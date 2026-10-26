/**
 * PHASE 44 regression test — when a hand is detected, the body's wrist
 * target must equal that hand's wristPos.
 *
 * Why this matters: the arm retargeter (bodyRetargeter.ts:454) aims the
 * forearm at `body.leftWrist`/`body.rightWrist`. The hand bone is placed at
 * `hand.wristPos`. Those come from two different MediaPipe measurements --
 * the POSE wrist landmarks (indices 15/16) vs the HAND GROUP's wrist
 * (whichever slot the hand-assignment resolver picks) -- and in every
 * recording in /avatar-data/ they disagree badly: the pose model
 * extrapolates the wrist from the shoulder/elbow line when the hand is
 * out of reach, shooting it up to ~5.7 shoulder-widths forward.
 *
 * With the divergence the avatar lunged at a point the hand never reached
 * -- the "going to another direction" symptom. The fix (mediapipeAdapter.ts)
 * overrides `body.wrist` with the detected hand's wristPos so the arm aims
 * at the hand.
 *
 * This test asserts the invariant directly on the recorded clips: for every
 * frame where a hand is detected, `body.wrist` MUST equal `hand.wristPos`.
 * A future refactor that re-introduces the divergence fails loudly.
 *
 * Run: npx tsx scripts/verify-wrist-aim.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import { LandmarkSmoother } from "../src/utils/landmarkSmoother";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const MEDIAPIPE_FRAME_FLOATS = 345;

/** Load a recorded clip as an array of 345-float buffers. */
function loadClip(label: string): Float32Array[] | null {
  const p = path.join(DATA_DIR, `${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`);
  if (!fs.existsSync(p)) return null;
  const data = JSON.parse(fs.readFileSync(p, "utf8"));
  const frames = data.frames;
  if (!Array.isArray(frames)) return null;
  if (frames.length > 0 && Array.isArray(frames[0])) {
    return frames.map((fr: number[]) => new Float32Array(fr));
  }
  const n = Math.floor(frames.length / MEDIAPIPE_FRAME_FLOATS);
  const out: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    out.push(new Float32Array(frames.slice(i * MEDIAPIPE_FRAME_FLOATS, (i + 1) * MEDIAPIPE_FRAME_FLOATS)));
  }
  return out;
}

const TARGETS = ["A", "HELLO", "ADOPT", "THANK_YOU", "ARE_YOU_DEAF"];
const EPS = 1e-3;

let fails = 0;
let checked = 0;
let handFrames = 0;
const smoother = new LandmarkSmoother(3, true, true);
for (const label of TARGETS) {
  const clip = loadClip(label);
  if (!clip) {
    console.log(`${label}: clip missing, skipped`);
    continue;
  }
  for (let i = 0; i < clip.length; i++) {
    const raw = clip[i];
    const buf = smoother.smooth(raw);
    const pose = mediapipeToCanonicalPose(buf, i / clip.length);
    if (!pose) continue;
    checked++;
    // The invariant: when a hand is detected, the body's wrist target MUST
    // be that hand's wristPos. The arm aims at body.wrist; the hand bone is
    // placed at hand.wristPos. If they disagree the avatar lunges.
    let leftErr = 0, rightErr = 0;
    if (pose.leftHand) {
      handFrames++;
      leftErr = Math.abs(pose.body.leftWrist[0] - pose.leftHand.wristPos[0])
              + Math.abs(pose.body.leftWrist[1] - pose.leftHand.wristPos[1])
              + Math.abs(pose.body.leftWrist[2] - pose.leftHand.wristPos[2]);
    }
    if (pose.rightHand) {
      handFrames++;
      rightErr = Math.abs(pose.body.rightWrist[0] - pose.rightHand.wristPos[0])
               + Math.abs(pose.body.rightWrist[1] - pose.rightHand.wristPos[1])
               + Math.abs(pose.body.rightWrist[2] - pose.rightHand.wristPos[2]);
    }
    if (leftErr > EPS) {
      console.log(`FAIL ${label} f${i}: left body.wrist=${JSON.stringify(pose.body.leftWrist)} != leftHand.wristPos=${JSON.stringify(pose.leftHand.wristPos)} (err=${leftErr.toFixed(4)})`);
      fails++;
    }
    if (rightErr > EPS) {
      console.log(`FAIL ${label} f${i}: right body.wrist=${JSON.stringify(pose.body.rightWrist)} != rightHand.wristPos=${JSON.stringify(pose.rightHand.wristPos)} (err=${rightErr.toFixed(4)})`);
      fails++;
    }
  }
}
console.log(`\nwrist-aim check: ${checked} frames, ${handFrames} hand detections across ${TARGETS.length} clips`);
console.log(fails === 0 ? "WRIST-AIM REGRESSION TEST PASSED" : `WRIST-AIM REGRESSION TEST FAILED: ${fails} violations`);
process.exit(fails === 0 ? 0 : 1);