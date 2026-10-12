/**
 * Palm-normal sign check.
 *
 * The hand retargeter curls fingers about an axis derived from
 * palmNormal, so if palmNormal points INTO the palm instead of
 * OUT, every finger curls backwards. The adapter uses
 *     right: cross(toPinky, toIndex)
 *     left:  cross(toIndex, toPinky)
 * claiming handedness symmetry (PHASE 15).
 *
 * Ground truth from the landmarks themselves: the palm
 * normal points toward the palm's CONCAVE side — the
 * side the fingers curl toward. The finger's curvature
 * (2nd difference of MCP/PIP/DIP) also points toward
 * the center of curvature, i.e. the SAME side. So
 *     dot(palmNormal, bend) > 0   = correct
 *     dot(palmNormal, bend) < 0   = inverted palm normal
 *
 * This is checked per side, per recording, over every frame with
 * a substantially bent finger.
 *
 * Run: npx tsx scripts/check-palmnormal.ts [recordings...]
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

/** palm normal exactly as computeHandPose builds it. */
function palmNormalOf(lms: Vec3[], side: "left" | "right"): Vec3 {
  const wrist = lms[0];
  const toIndex = sub(lms[5], wrist);
  const toPinky = sub(lms[17], wrist);
  return side === "right"
    ? norm(cross(toPinky, toIndex))
    : norm(cross(toIndex, toPinky));
}

/** The other order, for comparison. */
function palmNormalSwapped(lms: Vec3[], side: "left" | "right"): Vec3 {
  const wrist = lms[0];
  const toIndex = sub(lms[5], wrist);
  const toPinky = sub(lms[17], wrist);
  return side === "right"
    ? norm(cross(toIndex, toPinky))
    : norm(cross(toPinky, toIndex));
}

/** MCP/PIP/DIP indices per finger. */
const CHAINS = [
  [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19], [1, 2, 3],
];

function main() {
  const recordings =
    process.argv.slice(2).length > 0
      ? process.argv.slice(2)
      : ["FOUR", "FIVE", "THREE", "THANK_YOU", "KNOW", "I_LOVE_YOU", "HELLO"];

  // per side: [normalDot<0 count, normalDot>0 count, swappedDot<0, swappedDot>0]
  const agg: Record<string, [number, number, number, number]> = {
    left: [0, 0, 0, 0],
    right: [0, 0, 0, 0],
  };

  for (const rec of recordings) {
    const file = path.join(DATA_DIR, rec.endsWith(".json") ? rec : `${rec}.json`);
    if (!fs.existsSync(file)) {
      console.log(`!! ${rec}: not found`);
      continue;
    }
    const j = JSON.parse(fs.readFileSync(file, "utf8")) as { frames?: number[] };
    const flat = new Float32Array(j.frames ?? []);
    const frameCount = Math.floor(flat.length / FRAME_FLOATS);
    const perSide: Record<string, [number, number]> = { left: [0, 0], right: [0, 0] };

    for (let fi = 0; fi < frameCount; fi++) {
      const frame = flat.slice(fi * FRAME_FLOATS, (fi + 1) * FRAME_FLOATS);
      const pose = mediapipeToCanonicalPose(frame, fi / 24);
      if (!pose || !lastHandAssignment) continue;
      const assign = lastHandAssignment;

      for (const side of ["left", "right"] as const) {
        const off = side === "left" ? assign.left : assign.right;
        if (off === null) continue;
        const lms = readHand(frame, off);
        const pn = palmNormalOf(lms, side);
        const pnSwap = palmNormalSwapped(lms, side);

        for (const [m, p, d] of CHAINS) {
          const bend = sub(sub(lms[d], lms[p]), sub(lms[p], lms[m]));
          if (len(bend) < 0.05) continue;
        const a = agg[side];
        if (dot(pn, bend) > 0) { a[0]++; perSide[side][0]++; }
        else { a[1]++; perSide[side][1]++; }
        if (dot(pnSwap, bend) > 0) a[2]++;
        else a[3]++;
        }
      }
    }
    const L = perSide.left, R = perSide.right;
    const pct = (x: [number, number]) => (x[0] + x[1] === 0 ? "n/a" : `${(100 * x[1] / (x[0] + x[1])).toFixed(1)}%`);
    console.log(`${rec}: inverted-palmNormal rate  left ${pct(L)}   right ${pct(R)}`);
  }

  console.log("\n=== AGGREGATE (inverted palmNormal = fingers curl BACKWARD) ===");
  for (const side of ["left", "right"] as const) {
    const a = agg[side];
    const total = a[0] + a[1];
    const totalSwap = a[2] + a[3];
    console.log(
      `${side}:  current order inverted ${total ? (100 * a[1] / total).toFixed(1) : 0}% (${a[1]}/${total})` +
      `   swapped order inverted ${totalSwap ? (100 * a[3] / totalSwap).toFixed(1) : 0}% (${a[3]}/${totalSwap})`
    );
  }
}

main();
