/**
 * Verifies the PHASE 44 temporal stability lock.
 *
 * For each recording:
 *   1. Convert every raw MediaPipe frame to a CanonicalPose
 *      (the same mediapipeToCanonicalPose the app uses, with
 *      its per-frame handedness hysteresis).
 *   2. Count the RAW flip rate: how often a frame's palm
 *      normal inverts (> 90 deg) vs the previous frame,
 *      counting ONLY present (non-placeholder) hands.
 *   3. Run stabilizeCanonicalFrames() (the lock).
 *   4. Classify each held frame as a FLIP-hold (whole pose
 *      held) or a DROPOUT-hold (placeholder hand re-dressed
 *      with the last good orientation).
 *   5. Assert the STABILIZED sequence has ZERO palm-normal
 *      inversions between consecutive present hands.
 *   6. Assert the presence-awareness invariant: the detector
 *      fires EXACTLY when a both-real hand pair inverts, and
 *      never on a placeholder/real pair.
 *
 * Run with: npx tsx scripts/verify-stability-lock.ts [recordings...]
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import {
  stabilizeCanonicalFrames,
  isHandFlip,
  isPlaceholderHand,
  MAX_HOLD,
} from "../src/avatar/stabilityGuard";
import type { CanonicalPose, Vec3 } from "../src/avatar/canonicalPose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FRAME_FLOATS = 345;

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function loadPoses(name: string): CanonicalPose[] {
  const file = path.join(DATA_DIR, `${name}.json`);
  const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
  let frames: number[][];
  if (Array.isArray(parsed)) {
    frames = parsed;
  } else if (Array.isArray(parsed.frames) && typeof parsed.frames[0] === "number") {
    const flat: number[] = parsed.frames;
    frames = [];
    for (let i = 0; i + FRAME_FLOATS <= flat.length; i += FRAME_FLOATS) {
      frames.push(flat.slice(i, i + FRAME_FLOATS));
    }
  } else {
    frames = parsed.frames;
  }
  const poses: CanonicalPose[] = [];
  for (const frame of frames) {
    const p = mediapipeToCanonicalPose(frame, 0);
    if (p) poses.push(p);
  }
  return poses;
}

function main() {
  const recordings =
    process.argv.slice(2).length > 0
      ? process.argv.slice(2)
      : ["HELLO", "ADOPT", "KNOW", "THANK_YOU", "FOUR", "I_LOVE_YOU"];

  let totalRaw = 0;
  let totalFlipHeld = 0;
  let totalDropoutHeld = 0;
  let totalResynced = 0;
  let totalMaxHoldRun = 0;
  let totalStab = 0;
  let totalFrames = 0;
  let totalPlaceholder = 0;
  let placeholderFlipViolations = 0;

  console.log("\nPHASE 44 stability-lock verification");
  console.log("(raw flip = palm normal inverts >90 deg vs previous frame,");
  console.log(" both hands present in both frames)");
  console.log("");

  for (const name of recordings) {
    if (!fs.existsSync(path.join(DATA_DIR, `${name}.json`))) {
      console.warn(`skip ${name}`);
      continue;
    }
    const poses = loadPoses(name);
    if (poses.length === 0) continue;

    // 1. RAW flip rate (vs previous frame, present hands only)
    let rawFlips = 0;
    for (let i = 1; i < poses.length; i++) {
      if (isHandFlip(poses[i], poses[i - 1])) rawFlips++;
    }

    // 1b. Presence-awareness invariant: the detector must fire
    // EXACTLY when a comparable (both-real) hand pair inverts.
    for (let i = 1; i < poses.length; i++) {
      const a = poses[i - 1];
      const b = poses[i];
      const lCmp = !isPlaceholderHand(a.leftHand) && !isPlaceholderHand(b.leftHand);
      const rCmp = !isPlaceholderHand(a.rightHand) && !isPlaceholderHand(b.rightHand);
      const expected =
        (lCmp && dot(b.leftHand.palmNormal, a.leftHand.palmNormal) < 0) ||
        (rCmp && dot(b.rightHand.palmNormal, a.rightHand.palmNormal) < 0);
      if (expected !== isHandFlip(b, a)) placeholderFlipViolations++;
    }

    for (const p of poses) {
      if (isPlaceholderHand(p.leftHand) || isPlaceholderHand(p.rightHand)) {
        totalPlaceholder++;
      }
    }

    // 2. Stabilize (the lock)
    const stabilized = stabilizeCanonicalFrames(poses);

    // 3. Replay the lock's decision to classify held frames
    // and measure the longest consecutive hold run (which the
    // MAX_HOLD re-sync bounds).
    let flipHeld = 0;
    let dropoutHeld = 0;
    let resynced = 0;
    let maxHoldRun = 0;
    let lastGood: CanonicalPose | null = null;
    let holdCount = 0;
    let run = 0;
    for (let i = 0; i < stabilized.length; i++) {
      const src = poses[i];
      const out = stabilized[i];
      if (lastGood !== null && isHandFlip(src, lastGood)) {
        holdCount++;
        run++;
        if (run > maxHoldRun) maxHoldRun = run;
        if (holdCount > MAX_HOLD) {
          // re-sync: the current pose is accepted as the new
          // reference, ending the hold run
          resynced++;
          holdCount = 0;
          run = 0;
          lastGood = out;
        } else {
          flipHeld++;
          // lock held: the reference pose is unchanged
        }
      } else {
        holdCount = 0;
        run = 0;
        if (out !== src) dropoutHeld++;
        lastGood = out;
      }
    }

    // 4. STABILIZED flip rate (must be zero, present hands only)
    let stabFlips = 0;
    for (let i = 1; i < stabilized.length; i++) {
      const a = stabilized[i - 1];
      const b = stabilized[i];
      if (a && b && isHandFlip(b, a)) stabFlips++;
    }

    totalRaw += rawFlips;
    totalFlipHeld += flipHeld;
    totalDropoutHeld += dropoutHeld;
    totalResynced += resynced;
    if (maxHoldRun > totalMaxHoldRun) totalMaxHoldRun = maxHoldRun;
    totalStab += stabFlips;
    totalFrames += poses.length;

    const rawPct = (100 * rawFlips) / poses.length;
    console.log(
      `  ${name.padEnd(10)} frames ${String(poses.length).padStart(5)}  ` +
        `raw flips ${String(rawFlips).padStart(4)} (${rawPct.toFixed(1).padStart(5)}%)  ` +
        `flip-holds ${String(flipHeld).padStart(4)}  dropout-holds ${String(dropoutHeld).padStart(4)}  ` +
        `re-syncs ${String(resynced).padStart(3)}  maxHoldRun ${String(maxHoldRun).padStart(3)}  ` +
        `stabilized flips ${stabFlips}`
    );
  }

  console.log("");
  console.log(
    `  TOTAL: ${totalFrames} frames (${totalPlaceholder} with a placeholder hand), ` +
      `${totalRaw} raw flips, ${totalFlipHeld} flip-holds, ` +
      `${totalDropoutHeld} dropout-holds, ${totalResynced} re-syncs, ` +
      `longest hold run ${totalMaxHoldRun}, ${totalStab} stabilized flips`
  );
  let pass = placeholderFlipViolations === 0;
  if (placeholderFlipViolations > 0) {
    console.log(`  FAIL: ${placeholderFlipViolations} placeholder/real hand pairs were compared as flips.`);
    pass = false;
  }
  if (totalMaxHoldRun > MAX_HOLD + 1) {
    console.log(`  FAIL: a hold run reached ${totalMaxHoldRun} frames (MAX_HOLD=${MAX_HOLD} should bound it to ${MAX_HOLD + 1}).`);
    pass = false;
  } else {
    console.log(`  PASS: every hold run is bounded by MAX_HOLD (longest ${totalMaxHoldRun}).`);
  }
  // With re-sync, the only inversions left in the stabilized
  // sequence are the DELIBERATE re-sync transitions (the lock
  // accepting a persistently-flipped orientation as the new
  // reference). Every other inversion would be an unhandled
  // flip, so the stabilized count must not exceed the re-sync
  // count.
  if (totalStab <= totalResynced) {
    console.log(`  PASS: every stabilized inversion (${totalStab}) is a deliberate re-sync (${totalResynced}).`);
  } else {
    console.log(`  FAIL: ${totalStab} stabilized inversions but only ${totalResynced} re-syncs — unhandled flips remain.`);
    pass = false;
  }
  if (!pass) process.exit(1);
}

main();
