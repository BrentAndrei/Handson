/**
 * Handedness assignment audit.
 *
 * The palm normal's cross-product order is FIXED per
 * anatomical hand:
 *   LEFT  hand: cross(toIndex, toPinky)   points OUT of the palm
 *   RIGHT hand: cross(toPinky, toIndex)   points OUT of the palm
 *
 * The palm normal points toward the palm's CONCAVE
 * side — the side the fingers curl toward — and so
 * does the finger's curvature vector. So for a slot's
 * TRUE side, exactly one of the two cross orders
 * AGREES with the bend (dot > 0) — and that order
 * reveals the slot's true anatomical side, INDEPENDENT
 * of the geometric (nearest-pose-wrist) assignment.
 *
 * This script measures, per slot (99 = MediaPipe's "left"
 * slot, 162 = its "right" slot):
 *   - which anatomical side the slot really contains
 *     (from the bend test)
 *   - how often resolveHandAssignment's geometric choice
 *     disagrees with that ground truth
 *
 * Run: npx tsx scripts/check-handedness.ts [recordings...]
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { mediapipeToCanonicalPose, lastHandAssignment } from "../src/avatar/mediapipeAdapter";
import { mediapipeToCanonical } from "../src/avatar/coordinateSystem";
import type { Vec3 } from "../src/avatar/canonicalPose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FRAME_FLOATS = 345;
const SLOTS = { LEFT: 99, RIGHT: 162 };

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: Vec3): Vec3 => {
  const l = len(a);
  return l < 1e-10 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
};

function readHand(data: Float32Array, off: number): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < 21; i++) {
    const b = off + i * 3;
    out.push(mediapipeToCanonical([data[b], data[b + 1], data[b + 2]]) as Vec3);
  }
  return out;
}

const CHAINS = [[5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19], [1, 2, 3]];

/**
 * The palm normal points toward the palm's CONCAVE
 * side — the side the fingers curl toward — and so
 * does the finger's curvature vector. So the cross
 * order that AGREES with the bend (dot > 0) points
 * out of the palm, and reveals the slot's true
 * anatomical side:
 *   LEFT  hand: cross(toIndex, toPinky)   agrees
 *   RIGHT hand: cross(toPinky, toIndex)   agrees
 *
 * Returns [ltAgree, ltDisagree, rtAgree, rtDisagree]
 * where lt = cross(toIndex,toPinky) (the LEFT-hand
 * order) and rt = cross(toPinky,toIndex) (the
 * RIGHT-hand order).
 */
function sideEvidence(lms: Vec3[]): [number, number, number, number] {
  let ltA = 0, ltD = 0, rtA = 0, rtD = 0;
  const wrist = lms[0];
  const toIndex = sub(lms[5], wrist);
  const toPinky = sub(lms[17], wrist);
  const nLT = norm(cross(toIndex, toPinky));
  const nRT = norm(cross(toPinky, toIndex));
  for (const [m, p, d] of CHAINS) {
    const bend = sub(sub(lms[d], lms[p]), sub(lms[p], lms[m]));
    if (len(bend) < 0.05) continue;
    if (dot(nLT, bend) > 0) ltA++; else ltD++;
    if (dot(nRT, bend) > 0) rtA++; else rtD++;
  }
  return [ltA, ltD, rtA, rtD];
}

function main() {
  const recordings =
    process.argv.slice(2).length > 0
      ? process.argv.slice(2)
      : ["FOUR", "FIVE", "THREE", "THANK_YOU", "KNOW", "I_LOVE_YOU", "HELLO"];

  // per slot: frames where the bend test says LEFT, RIGHT, mixed, none
  const slotTruth: Record<string, { left: number; right: number; mixed: number; none: number }> = {
    "99": { left: 0, right: 0, mixed: 0, none: 0 },
    "162": { left: 0, right: 0, mixed: 0, none: 0 },
  };
  // assignment accuracy: chosen side vs bend-truth side
  const assignAcc = { correct: 0, wrong: 0, unknown: 0 };
  // per recording summary
  const perRec: string[] = [];

  // per recording, per slot: [leftVotes, rightVotes]
  const recSlot: Record<string, Record<string, [number, number]>> = {};

  for (const rec of recordings) {
    const file = path.join(DATA_DIR, rec.endsWith(".json") ? rec : `${rec}.json`);
    if (!fs.existsSync(file)) {
      perRec.push(`!! ${rec}: not found`);
      continue;
    }
    recSlot[rec] = { "99": [0, 0], "162": [0, 0] };
    const j = JSON.parse(fs.readFileSync(file, "utf8")) as { frames?: number[] };
    const flat = new Float32Array(j.frames ?? []);
    const frameCount = Math.floor(flat.length / FRAME_FLOATS);
    let recCorrect = 0, recWrong = 0, recUnknown = 0;

    for (let fi = 0; fi < frameCount; fi++) {
      const frame = flat.slice(fi * FRAME_FLOATS, (fi + 1) * FRAME_FLOATS);
      const pose = mediapipeToCanonicalPose(frame, fi / 24);
      if (!pose || !lastHandAssignment) continue;
      const assign = lastHandAssignment;

      // ground-truth side of each slot, from the bend test
      const truth: Record<number, "left" | "right" | "mixed" | "none"> = {};
      for (const off of Object.values(SLOTS)) {
        const key = String(off);
        // presence: spread test
        let cx = 0, cy = 0, cz = 0;
        for (let i = 0; i < 21; i++) {
          const b = off + i * 3;
          cx += frame[b]; cy += frame[b + 1]; cz += frame[b + 2];
        }
        cx /= 21; cy /= 21; cz /= 21;
        let spread = 0;
        for (let i = 0; i < 21; i++) {
          const b = off + i * 3;
          const dx = frame[b] - cx, dy = frame[b + 1] - cy, dz = frame[b + 2] - cz;
          spread = Math.max(spread, dx * dx + dy * dy + dz * dz);
        }
        if (Math.sqrt(spread) < 0.02) {
          truth[off] = "none";
          slotTruth[key].none++;
          continue;
        }
        const lms = readHand(frame, off);
        const [ltA, , rtA] = sideEvidence(lms);
        const ltWins = ltA > rtA;
        const rtWins = rtA > ltA;
        if (ltA > 0 && rtA > 0 && ltA === rtA) {
          truth[off] = "mixed";
          slotTruth[key].mixed++;
        } else if (ltWins && !rtWins) {
          truth[off] = "left";
          slotTruth[key].left++;
          recSlot[rec][key][0]++;
        } else if (rtWins && !ltWins) {
          truth[off] = "right";
          slotTruth[key].right++;
          recSlot[rec][key][1]++;
        } else {
          truth[off] = "mixed";
          slotTruth[key].mixed++;
        }
      }

      // assignment vs truth
      for (const side of ["left", "right"] as const) {
        const chosen = side === "left" ? assign.left : assign.right;
        if (chosen === null) continue;
        const t = truth[chosen];
        if (t === "left" || t === "right") {
          if (t === side) { assignAcc.correct++; recCorrect++; }
          else { assignAcc.wrong++; recWrong++; }
        } else {
          assignAcc.unknown++; recUnknown++;
        }
      }
    }
    perRec.push(
      `${rec}: assignment correct ${recCorrect}  wrong ${recWrong}  unknown ${recUnknown}`
    );
  }

  console.log("=== PER-RECORDING SLOT TRUTH (bend votes: L vs R) ===");
  for (const [rec, slots] of Object.entries(recSlot)) {
    for (const [slot, [l, r]] of Object.entries(slots)) {
      const tot = l + r;
      if (tot === 0) continue;
      console.log(
        `  ${rec} slot ${slot}: LEFT ${l}  RIGHT ${r}  (${(100 * Math.max(l, r) / tot).toFixed(0)}% dominant)`
      );
    }
  }

  console.log("=== SLOT TRUE SIDE (bend test, assignment-independent) ===");
  for (const [name, s] of Object.entries(slotTruth)) {
    const tot = s.left + s.right + s.mixed + s.none;
    const pres = s.left + s.right + s.mixed;
    console.log(
      `slot ${name}: present ${pres}/${tot}  ` +
      `LEFT ${pres ? (100 * s.left / pres).toFixed(1) : 0}%  ` +
      `RIGHT ${pres ? (100 * s.right / pres).toFixed(1) : 0}%  ` +
      `mixed ${pres ? (100 * s.mixed / pres).toFixed(1) : 0}%`
    );
  }
  console.log("\n=== GEOMETRIC ASSIGNMENT vs BEND-TRUTH ===");
  const tot = assignAcc.correct + assignAcc.wrong;
  console.log(
    `correct ${assignAcc.correct}  wrong ${assignAcc.wrong}  ` +
    `(${tot ? (100 * assignAcc.wrong / tot).toFixed(1) : 0}% wrong)  unknown ${assignAcc.unknown}`
  );
  console.log("\n" + perRec.join("\n"));
}

main();
