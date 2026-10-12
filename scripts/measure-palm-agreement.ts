/**
 * Measures whether the palmNormal sign (which depends on the
 * handedness assignment) agrees with the ground-truth finger
 * bend direction (bendDir, the 2nd difference of MCP/PIP/DIP).
 *
 * If the assignment is reliable, bendDir points INTO the palm,
 * i.e. along -palmNormal. dot(bendDir, -palmNormal) > 0.
 *
 * A high agreement rate means -palmNormal is a ROBUST bend
 * direction (computed from the whole palm triangle), which is
 * far less noisy than the per-finger 2nd difference that the
 * current Phase 37 sign correction uses.
 *
 * Run with: npx tsx scripts/measure-palm-agreement.ts [recordings...]
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import type { Vec3 } from "../src/avatar/canonicalPose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FRAME_FLOATS = 345;

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

type FingerKey = "thumb" | "index" | "middle" | "ring" | "pinky";
const FINGERS: FingerKey[] = ["index", "middle", "ring", "pinky", "thumb"];

function main() {
  const recordings =
    process.argv.slice(2).length > 0
      ? process.argv.slice(2)
      : ["FOUR", "FIVE", "THREE", "KNOW", "THANK_YOU", "I_LOVE_YOU"];

  // per hand, per finger: { agree, disagree, nobend }
  const stats: Record<string, Record<FingerKey, { agree: number; disagree: number; nobend: number }>> = {
    left: { thumb: { agree: 0, disagree: 0, nobend: 0 }, index: { agree: 0, disagree: 0, nobend: 0 }, middle: { agree: 0, disagree: 0, nobend: 0 }, ring: { agree: 0, disagree: 0, nobend: 0 }, pinky: { agree: 0, disagree: 0, nobend: 0 } },
    right: { thumb: { agree: 0, disagree: 0, nobend: 0 }, index: { agree: 0, disagree: 0, nobend: 0 }, middle: { agree: 0, disagree: 0, nobend: 0 }, ring: { agree: 0, disagree: 0, nobend: 0 }, pinky: { agree: 0, disagree: 0, nobend: 0 } },
  };

  for (const name of recordings) {
    const file = path.join(DATA_DIR, `${name}.json`);
    if (!fs.existsSync(file)) { console.warn(`skip ${name}`); continue; }
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    // Frames may be a nested array OR one flat Float32-style array
    // (345 floats per frame, concatenated).
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

    for (const frame of frames) {
      const pose = mediapipeToCanonicalPose(frame, 0);
      if (!pose) continue;

      for (const side of ["left", "right"] as const) {
        const hand = side === "left" ? pose.leftHand : pose.rightHand;
        if (!hand) continue;
        const palmNormal: Vec3 = [hand.palmNormal[0], hand.palmNormal[1], hand.palmNormal[2]];
        const negPalm: Vec3 = [-palmNormal[0], -palmNormal[1], -palmNormal[2]];

        for (const finger of FINGERS) {
          const fp = hand.fingers[finger];
          if (!fp) continue;
          const s = stats[side][finger];
          if (!fp.bendDir || fp.curl < 0.05) { s.nobend++; continue; }
          const d = dot(fp.bendDir, negPalm);
          if (d > 0) s.agree++; else s.disagree++;
        }
      }
    }
  }

  console.log("\npalmNormal-sign agreement with ground-truth bendDir");
  console.log("(dot(bendDir, -palmNormal) > 0 means the assignment is correct)");
  console.log("");
  let totAgree = 0, totDisagree = 0;
  for (const side of ["left", "right"] as const) {
    console.log(`  ${side}:`);
    for (const finger of FINGERS) {
      const s = stats[side][finger];
      const total = s.agree + s.disagree;
      totAgree += s.agree; totDisagree += s.disagree;
      const rate = total > 0 ? (100 * s.agree) / total : 0;
      const flip = total > 0 ? (100 * s.disagree) / total : 0;
      console.log(`    ${finger.padEnd(7)} agree ${rate.toFixed(1).padStart(5)}%   flip ${flip.toFixed(1).padStart(5)}%   (n=${total})`);
    }
  }
  const grand = totAgree + totDisagree;
  console.log(`\n  OVERALL agreement: ${(100 * totAgree) / grand}%   implied flip if using -palmNormal: ${(100 * totDisagree) / grand}%`);
}

main();
