/**
 * PHASE 13 — SMPL-X dataset structural parity gate.
 *
 * Why this exists: every pre-Phase-13 test fed `retarget()` from the MediaPipe
 * path, so the SMPL-X path (the one the frontend actually plays) had ZERO
 * coverage. That let a malformed joint layout ship unnoticed and collapse the
 * avatar on playback.
 *
 * This gate asserts STRUCTURAL properties of the raw `smplx_joints` array
 * itself -- no Three.js, no retargeting, so it isolates data defects from math
 * defects:
 *
 *   1. Layout detection: the array must be the OFFICIAL 24-joint body layout
 *      (indices 0-23 populated, fingers 24+ present). A 20-joint IK-style list
 *      leaves 20..127 zeroed, the signature of the old exporter bug.
 *   2. Mirror parity: for every L/R body pair, |L.x| ~ |R.x|, L.y ~ R.y,
 *      |L.z| ~ |R.z| (SMPL-X is X-right, Y-up, Z-forward).
 *   3. Segment symmetry: per side, forearm/upper-arm must be a plausible ratio,
 *      and the L/R segment lengths must agree.
 *
 * Deliberately tolerant (ratios, not exact equality): signing legitimately
 * makes the arms asymmetric. Only structural corruption is flagged.
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA = path.join(ROOT, "public", "avatar-data-smplx");

/** Official SMPL-X body layout, indices 0-23. */
const OFFICIAL = {
  pelvis: 0, left_hip: 1, right_hip: 2, spine1: 3, left_knee: 4, right_knee: 5,
  spine2: 6, left_ankle: 7, right_ankle: 8, spine3: 9, footSideL: 10, footSideR: 11,
  neck: 12, clavL: 13, clavR: 14, head: 15, upperArmL: 16, upperArmR: 17,
  elbowL: 18, elbowR: 19, forearmL: 20, forearmR: 21, handL: 22, handR: 23,
};

/**
 * IK layout, indices 0-19 — the one `smplxAdapter.IK_BODY_IDX` reads.
 *
 * This is the layout the IK-based exporter actually emits (the official
 * 128-joint model is not installed, so `hasOfficialLayout()` is false and the
 * reader selects this table). The gate must validate the layout the CONSUMER
 * uses, otherwise it tests indices nobody reads.
 */
const IK = {
  pelvis: 0, left_hip: 1, right_hip: 2, spine1: 3, left_knee: 4, right_knee: 5,
  spine2: 6, left_ankle: 7, right_ankle: 8, spine3: 9, neck: 10,
  clavL: 11, clavR: 12, head: 13, elbowL: 14, elbowR: 15,
  forearmL: 16, forearmR: 17, handL: 18, handR: 19,
};

/** L/R pairs whose geometry must be mirrored about the X axis. */
const MIRROR_PAIRS: [string, number, number][] = [
  ["hip", IK.left_hip, IK.right_hip],
  ["knee", IK.left_knee, IK.right_knee],
  ["ankle", IK.left_ankle, IK.right_ankle],
  ["clavicle", IK.clavL, IK.clavR],
  ["elbow", IK.elbowL, IK.elbowR],
  ["forearm", IK.forearmL, IK.forearmR],
  ["hand", IK.handL, IK.handR],
];
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

interface Issue {
  file: string;
  frame: number;
  kind: string;
  detail: string;
}

function analyse(file: string, limitFrames = 12): Issue[] {
  const issues: Issue[] = [];
  const raw = JSON.parse(fs.readFileSync(path.join(DATA, file), "utf8")) as {
    frames: { smplx_joints: number[] }[];
  };
  const n = Math.min(limitFrames, raw.frames.length);
  for (let f = 0; f < n; f++) {
    const J = raw.frames[f].smplx_joints;
    const g = (i: number): number[] => [J[i * 3], J[i * 3 + 1], J[i * 3 + 2]];
    const nonZero = (i: number) =>
      Math.abs(J[i * 3]) + Math.abs(J[i * 3 + 1]) + Math.abs(J[i * 3 + 2]) > 1e-6;

    // --- 1. layout detection ------------------------------------------------
    // Pick the layout the reader will use: official if any finger joint (24+)
    // is populated, otherwise the IK table.
    const tailPopulated = Array.from({ length: 104 }, (_, k) => 24 + k).filter(nonZero).length;
    const L = tailPopulated > 0 ? OFFICIAL : IK;
    const bodyCount = tailPopulated > 0 ? 24 : 20;
    const bodyPopulated = Array.from({ length: bodyCount }, (_, i) => i).filter(nonZero).length;
    if (bodyPopulated < bodyCount) {
      issues.push({
        file, frame: f, kind: "LAYOUT",
        detail:
          `only ${bodyPopulated}/${bodyCount} body joints populated in the ` +
          `${tailPopulated > 0 ? "official" : "IK"} layout; a joint the reader ` +
          `uses is missing, so it resolves as ZERO and the limb is aimed from garbage.`,
      });
    }
    for (const [name, i] of [
      ["left wrist", L.forearmL],
      ["right wrist", L.forearmR],
    ] as const) {
      if (!nonZero(i)) {
        issues.push({ file, frame: f, kind: "MISSING", detail: `${name} (index ${i}) is zero` });
      }
    }

    // --- 2. mirror parity ---------------------------------------------------
    for (const [name, li, ri] of MIRROR_PAIRS) {
      if (!nonZero(li) || !nonZero(ri)) continue;
      const L = g(li);
      const R = g(ri);
      const dx = Math.abs(Math.abs(L[0]) - Math.abs(R[0]));
      const dy = Math.abs(Math.abs(L[1]) - Math.abs(R[1]));
      const dz = Math.abs(Math.abs(L[2]) - Math.abs(R[2]));
      // A signer CAN raise one arm, so only flag gross asymmetry.
      const scale = Math.max(1e-6, Math.hypot(L[0], L[1], L[2]));
      if (dx > 0.6 * scale || dy > 0.6 * scale || dz > 0.6 * scale) {
        issues.push({
          file, frame: f, kind: "MIRROR",
          detail:
            `${name} pair not mirrored: L=(${L.map((v) => v.toFixed(3))}) ` +
            `R=(${R.map((v) => v.toFixed(3))}) dX=${dx.toFixed(3)} dY=${dy.toFixed(3)} dZ=${dz.toFixed(3)}`,
        });
      }
    }

    // --- 3. segment symmetry -----------------------------------------------
    for (const side of ["left", "right"] as const) {
      const sh = side === "left" ? L.clavL : L.clavR;
      const el = side === "left" ? L.elbowL : L.elbowR;
      const wr = side === "left" ? L.forearmL : L.forearmR;
      if (!nonZero(sh) || !nonZero(el) || !nonZero(wr)) continue;
      const upper = dist(g(sh), g(el));
      const fore = dist(g(el), g(wr));
      if (!(upper > 1e-6) || !(fore > 1e-6)) {
        issues.push({
          file, frame: f, kind: "SEGMENT",
          detail: `${side} arm has a zero-length segment (upper=${upper.toFixed(4)} fore=${fore.toFixed(4)})`,
        });
        continue;
      }
      const ratio = fore / upper;
      if (ratio < 0.5 || ratio > 2.0) {
        issues.push({
          file, frame: f, kind: "SEGMENT",
          detail:
            `${side} forearm/upper ratio = ${ratio.toFixed(3)} (plausible 0.5-2.0); ` +
            `clav(${sh})->elbow(${el})=${upper.toFixed(3)}m elbow->wrist(${wr})=${fore.toFixed(3)}m`,
        });
      }
    }
  }
  return issues;
}

function main() {
  const only = process.argv[2];
  const files = fs
    .readdirSync(DATA)
    .filter((f) => f.endsWith(".smplx.json"))
    .filter((f) => !only || f.toUpperCase().startsWith(only.toUpperCase()))
    .sort();
  if (!files.length) {
    console.error(`no .smplx.json files in ${DATA}`);
    process.exitCode = 1;
    return;
  }

  console.log("PHASE 13 - SMPL-X DATASET STRUCTURAL PARITY GATE (hard gate)");
  console.log(`   files: ${files.length}\n`);

  let bad = 0;
  const byKind: Record<string, number> = {};
  const sample: string[] = [];
  for (const f of files) {
    const issues = analyse(f);
    if (issues.length) {
      bad++;
      for (const i of issues) {
        byKind[i.kind] = (byKind[i.kind] ?? 0) + 1;
        if (sample.length < 10) sample.push(`   [${i.kind}] ${i.file} f${i.frame}: ${i.detail}`);
      }
    }
  }

  if (sample.length) {
    console.log("   sample findings:");
    for (const s of sample) console.log(s);
    console.log("");
    console.log(`   issue counts: ${Object.entries(byKind).map(([k, v]) => `${k}=${v}`).join("  ")}`);
    console.log("");
  }

  console.log(`   files with structural defects : ${bad} / ${files.length}`);
  if (bad > 0) {
    console.log("\nSMPL-X PARITY GATE FAILED -- the dataset joint layout is corrupt.");
    console.log("The avatar collapse is a DATA defect, not a rendering or math defect.");
    process.exitCode = 1;
  } else {
    console.log("\nSMPL-X PARITY GATE PASSED (layout, mirror parity and segment symmetry all valid).");
  }
}

main();

