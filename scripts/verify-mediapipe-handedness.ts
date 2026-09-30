/**
 * PHASE 3 (E8) — MediaPipe -> Canonical handedness verification.
 *
 * Deterministic synthetic MediaPipe landmark input: a signer FACING the camera
 * in a raw, UNMIRRORED frame (the app's actual capture path: ctx.drawImage ->
 * detectForVideo -> packHolisticResult -> normalizeLandmarks — no flip anywhere).
 * For a camera-facing signer the anatomical LEFT is on the image RIGHT
 * (x grows toward the image's right edge).
 *
 * Pipeline under test:
 *   packed buffer -> normalizeLandmarks() -> mediapipeToCanonicalPose()
 *
 * Assertions are made against the project canonical convention
 *   X = forward (toward viewer), Y = up, Z = SUBJECT'S RIGHT (subject's left = -Z)
 * — the same convention verify-retarget.ts pins against the Xbot bind pose
 * (tposeJoints: lSh z=-0.20, rSh z=+0.20, "subject's left = -Z_canonical").
 *
 * Any accidental mirror in coordinateSystem.mediapipeToCanonical() surfaces as
 * left/right-swapped Z coordinates and a left arm that points toward +Z.
 *
 * Run with: npx tsx scripts/verify-mediapipe-handedness.ts
 *           npm run verify:handedness
 *
 * Writes verify-handedness.json (machine-readable, for before/after diffs).
 * Exit code 1 if any check fails (usable as a regression test).
 */
import * as fs from "fs";
import { fileURLToPath } from "url";
import {
  POSE_OFFSET,
  TOTAL_FLOATS,
  type LandmarksResult,
} from "../src/mediapipe/types";
import { normalizeLandmarks } from "../src/utils/normalizeLandmarks";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import {
  mediapipeToCanonical,
  mediapipeArrayToCanonical,
} from "../src/avatar/coordinateSystem";
import type { BodyPose, Vec3 } from "../src/avatar/canonicalPose";

// ---------------------------------------------------------------------------
// Synthetic input: raw MediaPipe image coordinates of a camera-facing signer.
//   x: 0 = image left edge .. 1 = image right edge
//   y: 0 = image top      .. 1 = image bottom
//   z: +z = away from camera (negative = closer to camera), origin ~ hip center
// Landmark indices per MediaPipe Pose (11/12 = anatomical left/right shoulder).
// ---------------------------------------------------------------------------
type Landmark = { x: number; y: number; z: number };
const lm = (x: number, y: number, z = 0): Landmark => ({ x, y, z });

const SIGNER_FACING_CAMERA: Record<number, Landmark> = {
  0: lm(0.50, 0.22, -0.05), // nose (closer to camera than shoulders)
  11: lm(0.65, 0.35), // LEFT  shoulder -> image RIGHT half
  12: lm(0.35, 0.35), // RIGHT shoulder -> image LEFT  half
  13: lm(0.80, 0.35), // LEFT  elbow    -> further right (arm out to subject's left)
  14: lm(0.30, 0.50), // RIGHT elbow    -> further left, lower (arm down+out)
  15: lm(0.95, 0.35), // LEFT  wrist    -> far image right (left arm extended sideways)
  16: lm(0.42, 0.62), // RIGHT wrist    -> forearm folds back inward/down
  23: lm(0.60, 0.62), // LEFT  hip
  24: lm(0.40, 0.62), // RIGHT hip
  25: lm(0.62, 0.80), // LEFT  knee
  26: lm(0.38, 0.80), // RIGHT knee
  27: lm(0.63, 0.95), // LEFT  ankle
  28: lm(0.37, 0.95), // RIGHT ankle
};

/** Pack the synthetic pose into the app's shared landmark layout (pose only). */
function packFrame(): LandmarksResult {
  const buffer = new ArrayBuffer(TOTAL_FLOATS * 4);
  const view = new Float32Array(buffer); // zero-filled: absent groups stay absent
  for (const key of Object.keys(SIGNER_FACING_CAMERA)) {
    const idx = Number(key);
    const p = SIGNER_FACING_CAMERA[idx];
    const base = POSE_OFFSET + idx * 3;
    view[base] = p.x;
    view[base + 1] = p.y;
    view[base + 2] = p.z;
  }
  return {
    timestamp: 0,
    buffer,
    hasPose: true,
    hasLeftHand: false,
    hasRightHand: false,
    hasFace: false,
  };
}

// ---------------------------------------------------------------------------
// Reporting (same shape as verify-retarget.ts)
// ---------------------------------------------------------------------------
type Check = {
  name: string;
  expected: string;
  actual: string;
  pass: boolean;
  note?: string;
};
const CHECKS: Check[] = [];

const fmt = (v: Vec3) =>
  `[${v[0].toFixed(4)}, ${v[1].toFixed(4)}, ${v[2].toFixed(4)}]`;
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l < 1e-12 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
};
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

function check(name: string, pass: boolean, expected: string, actual: string, note?: string) {
  CHECKS.push({ name, expected, actual, pass, note });
  console.log(
    `  [${pass ? "ok  " : "FAIL"}] ${name.padEnd(42)} expected=${expected} actual=${actual}` +
      (note && !pass ? `  <- ${note}` : "")
  );
}

/** Absolute per-component position match (deterministic input => exact values). */
function checkPos(name: string, got: Vec3, want: Vec3, tol = 1e-5) {
  const pass = Math.abs(got[0] - want[0]) <= tol &&
    Math.abs(got[1] - want[1]) <= tol &&
    Math.abs(got[2] - want[2]) <= tol;
  const note = pass ? undefined : dot(norm(got), norm(want)) < 0
    ? "direction is MIRRORED (opposite side)"
    : "position mismatch";
  check(name, pass, fmt(want), fmt(got), note);
}

/** Direction match by dot product: a mirror gives dot = -1 -> always caught. */
function checkDir(name: string, got: Vec3, want: Vec3, minDot = 0.9999) {
  const d = dot(norm(got), norm(want));
  check(
    name,
    d >= minDot,
    `${fmt(want)} dot>=${minDot}`,
    `${fmt(norm(got))} dot=${d.toFixed(4)}`,
    d < 0 ? "points the OPPOSITE way (left/right mirror)" : undefined
  );
}
// ---------------------------------------------------------------------------
// Expected canonical values (derivation):
//   anchor = shoulder midpoint = (0.5, 0.35, 0);  scale = shoulder width = 0.30
//   normalized left shoulder = ((0.65-0.5)/0.3, 0, 0) = (+0.5, 0, 0)
//   fixed map canon = (-z, -y, -x) must land the subject's left on -Z:
//     subject's left shoulder  -> [0, 0, -0.5]   (-Z = subject's left side)
//     subject's right shoulder -> [0, 0, +0.5]   (+Z = subject's right side)
// ---------------------------------------------------------------------------
const E_LSH: Vec3 = [0, 0, -0.5];
const E_RSH: Vec3 = [0, 0, 0.5];
const E_LEL: Vec3 = [0, 0, -1.0]; // left arm extended out to the subject's left
const E_LWR: Vec3 = [0, 0, -1.5];
const E_REL: Vec3 = [0, -0.5, 2 / 3]; // (0.30,0.50) norm -> (-2/3, 1/2)
const E_RWR: Vec3 = [0, -0.9, 0.08 / 0.3]; // (0.42,0.62) norm -> (-0.08/0.3, 0.9)
const E_LUPPER: Vec3 = [0, 0, -1];
const E_LFORE: Vec3 = [0, 0, -1];
const E_RUPPER: Vec3 = [0, -0.5, 2 / 3 - 0.5]; // rEl - rSh = (0,-0.5,1/6); checkDir normalizes
const E_RFORE: Vec3 = [0, -1 / Math.SQRT2, -1 / Math.SQRT2]; // normalize([0,-0.4,-0.4])
const E_AXIS: Vec3 = [0, 0, 1]; // shoulderAxis = L->R = subject's right
const E_SPINE: Vec3 = [0, 1, 0];

function main() {
  console.log("########## E8: MediaPipe -> Canonical handedness ##########\n");

  // --- Q4/Q6 unit check: the conversion must be a ROTATION (det = +1).
  // Camera-facing subject in MediaPipe image coords:
  //   forward = toward camera = (0,0,-1); up = (0,-1,0);
  //   subject's right = image left = (-1,0,0).
  // Right-handed canonical frame requires: forward x up == subject's right.
  console.log("--- 1. mediapipeToCanonical basis chirality (Q4/Q6) ---");
  {
    const f = mediapipeToCanonical([0, 0, -1]);
    const u = mediapipeToCanonical([0, -1, 0]);
    const r = mediapipeToCanonical([-1, 0, 0]);
    checkPos("basis: forward -> +X", f, [1, 0, 0], 1e-9);
    checkPos("basis: up -> +Y", u, [0, 1, 0], 1e-9);
    checkPos("basis: subject-right -> +Z", r, [0, 0, 1], 1e-9);
    checkDir("right-handed: fwd x up == right", cross(f, u), r, 0.9999);

    // array variant must be the same conversion pointwise
    const raw: number[] = [0, 0, -1, 0, -1, 0, -1, 0, 0];
    const arr = mediapipeArrayToCanonical(raw); // THREE.Vector3[]
    let agree = true;
    for (let i = 0; i < 3; i++) {
      const one = mediapipeToCanonical([raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]]);
      const got: Vec3 = [arr[i].x, arr[i].y, arr[i].z];
      if (![0, 1, 2].every((c) => Math.abs(got[c] - one[c]) < 1e-9)) agree = false;
    }
    check("mediapipeArrayToCanonical agrees pointwise", agree, "true", String(agree));
  }

  // --- Full pipeline: packed buffer -> normalizeLandmarks -> canonical pose
  console.log("\n--- 2. synthetic frame through the real pipeline ---");
  const normalized = normalizeLandmarks(packFrame());
  const pose = mediapipeToCanonicalPose(normalized.vector, 0);
  if (!pose) {
    check("mediapipeToCanonicalPose returns a pose", false, "non-null", "null");
    return report();
  }
  check("mediapipeToCanonicalPose returns a pose", true, "non-null", "non-null");
  const b: BodyPose = pose.body;

  console.log("\n--- 3. left/right shoulder positions (Q3/Q8/Q9) ---");
  checkPos("leftShoulder position", b.leftShoulder, E_LSH);
  checkPos("rightShoulder position", b.rightShoulder, E_RSH);
  check(
    "leftShoulder on -Z, rightShoulder on +Z",
    b.leftShoulder[2] < 0 && b.rightShoulder[2] > 0,
    "L.z<0<R.z",
    `L.z=${b.leftShoulder[2].toFixed(4)} R.z=${b.rightShoulder[2].toFixed(4)}`,
    b.leftShoulder[2] > 0 ? "left/right SWAPPED" : undefined
  );

  console.log("\n--- 4. left/right elbow positions ---");
  checkPos("leftElbow position", b.leftElbow, E_LEL);
  checkPos("rightElbow position", b.rightElbow, E_REL);
  check(
    "leftElbow further left than leftShoulder",
    b.leftElbow[2] < b.leftShoulder[2],
    "z < -0.5",
    `z=${b.leftElbow[2].toFixed(4)}`
  );
  check(
    "rightElbow further right than rightShoulder",
    b.rightElbow[2] > b.rightShoulder[2],
    "z > 0.5",
    `z=${b.rightElbow[2].toFixed(4)}`
  );

  console.log("\n--- 5. left/right wrist positions ---");
  checkPos("leftWrist position", b.leftWrist, E_LWR);
  checkPos("rightWrist position", b.rightWrist, E_RWR);
  check(
    "leftWrist on -Z, rightWrist on +Z",
    b.leftWrist[2] < 0 && b.rightWrist[2] > 0,
    "L.z<0<R.z",
    `L.z=${b.leftWrist[2].toFixed(4)} R.z=${b.rightWrist[2].toFixed(4)}`,
    b.leftWrist[2] > 0 ? "left/right SWAPPED" : undefined
  );
  console.log("\n--- 6. left/right upper-arm directions (Q8/Q9) ---");
  const lUpper = norm(sub(b.leftElbow, b.leftShoulder));
  const rUpper = norm(sub(b.rightElbow, b.rightShoulder));
  checkDir("left upper-arm dir", sub(b.leftElbow, b.leftShoulder), E_LUPPER);
  checkDir("right upper-arm dir", sub(b.rightElbow, b.rightShoulder), E_RUPPER);
  check(
    "canonical left arm points to -Z (subject's left)",
    dot(lUpper, [0, 0, -1]) >= 0.9999,
    "dot((0,0,-1)) >= 0.9999",
    `dot=${dot(lUpper, [0, 0, -1]).toFixed(4)}`,
    dot(lUpper, [0, 0, -1]) < 0 ? "left arm points toward canonical RIGHT" : undefined
  );
  check(
    "canonical right arm has +Z component (subject's right)",
    rUpper[2] > 0,
    "z > 0",
    `z=${rUpper[2].toFixed(4)}`,
    rUpper[2] < 0 ? "right arm points toward canonical LEFT" : undefined
  );

  console.log("\n--- 7. left/right forearm directions ---");
  checkDir("left forearm dir", sub(b.leftWrist, b.leftElbow), E_LFORE);
  checkDir("right forearm dir", sub(b.rightWrist, b.rightElbow), E_RFORE, 0.999);
  check(
    "right forearm folds inward (wrist Z < elbow Z)",
    b.rightWrist[2] < b.rightElbow[2],
    "z < 0.6667",
    `z=${b.rightWrist[2].toFixed(4)}`
  );

  console.log("\n--- 8. frame axes (mirror + facing sanity) ---");
  checkDir("shoulderAxis L->R == +Z", b.shoulderAxis, E_AXIS, 0.9999);
  checkDir("spineDir == +Y", b.spineDir, E_SPINE, 0.9999);
  check("head faces +X (nose toward camera)", b.headDir[0] > 0, "x > 0", `x=${b.headDir[0].toFixed(4)}`);
  check(
    "shoulderMid above hipMid (Y up)",
    b.shoulderHeight > b.hipsPos[1],
    "shoulderHeight > hips.y",
    `sh=${b.shoulderHeight.toFixed(4)} hips.y=${b.hipsPos[1].toFixed(4)}`
  );

  return report();
}

function report() {
  const failed = CHECKS.filter((c) => !c.pass);
  console.log("\n########## SUMMARY ##########");
  console.log(`checks: ${CHECKS.length - failed.length}/${CHECKS.length} passed`);
  if (failed.length) {
    console.log("failing checks:");
    for (const c of failed) console.log(`  - ${c.name}: expected=${c.expected} actual=${c.actual}`);
    console.log("\nE8 VERDICT: handedness NOT preserved -> coordinate map mirrors the subject.");
  } else {
    console.log("\nE8 VERDICT: handedness preserved (subject's left = -Z, right = +Z).");
  }

  fs.writeFileSync(
    fileURLToPath(new URL("../verify-handedness.json", import.meta.url)),
    JSON.stringify(
      {
        convention: "X=forward(toward viewer), Y=up, Z=subject's right; subject's left = -Z",
        input: "synthetic camera-facing signer, raw unmirrored image coords",
        checks: CHECKS,
      },
      null,
      2
    )
  );
  console.log("wrote verify-handedness.json");
  process.exit(failed.length ? 1 : 0);
}

main();
