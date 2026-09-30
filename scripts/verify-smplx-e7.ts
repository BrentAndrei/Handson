/**
 * PHASE 6 — E7 end-to-end impact on the rendered avatar.
 *
 * E7 is a bug in the PYTHON fitter (signavatars-service/fit_smplx.py): the
 * world->local quaternion conversion used its operands in the wrong order, so
 * every joint below a rotated parent got a wrong LOCAL rotation. Proven and
 * fixed there, and it changed real output by up to ~90 deg per joint.
 *
 * This script answers the question that actually matters: does that wrong
 * local rotation change what the avatar RENDERS? The TS adapter
 * (smplxAdapter.ts) can source a frame's body either from `smplx_joints`
 * (3D joint POSITIONS, produced by the IK solver before the buggy conversion
 * ran) or from `pose` (the axis-angle parameters the buggy conversion wrote).
 * If positions drive the body, the shipped avatar was largely immune and E7 was
 * a latent data-correctness bug; if `pose` drives it, E7 was live.
 *
 * Loads a BUGGY-generated and a FIXED-generated .smplx.json for the same source
 * clip, runs BOTH through the real smplxPoseToCanonicalPose(), and compares the
 * resulting canonical body directions.
 *
 * Run:  npx tsx scripts/verify-smplx-e7.ts <buggyDir> <fixedDir>
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { smplxPoseToCanonicalPose } from "../src/avatar/smplxAdapter";
import { SMPLX_JOINT_COUNT, type SmplxPose } from "../src/avatar/smplxConverter";
import { canonicalToGltf, type Coords } from "../src/avatar/coordinateSystem";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

type Vec3 = [number, number, number];

const gltf = (v: Vec3) => {
  const g = canonicalToGltf(v as Coords);
  return new THREE.Vector3(g[0], g[1], g[2]);
};

interface DirCase {
  name: string;
  a: (b: any) => THREE.Vector3;
  b: (b: any) => THREE.Vector3;
}

/** Body directions, expressed off the canonical BodyPose. */
const DIR_CASES: DirCase[] = [
  { name: "leftUpperArm", a: (b) => gltf(b.leftElbow).sub(gltf(b.leftShoulder)), b: (b) => gltf(b.leftWrist).sub(gltf(b.leftElbow)) },
  { name: "rightUpperArm", a: (b) => gltf(b.rightElbow).sub(gltf(b.rightShoulder)), b: (b) => gltf(b.rightWrist).sub(gltf(b.rightElbow)) },
  { name: "leftThigh", a: (b) => gltf(b.leftKnee).sub(gltf(b.leftHip)), b: (b) => gltf(b.leftAnkle).sub(gltf(b.leftKnee)) },
  { name: "rightThigh", a: (b) => gltf(b.rightKnee).sub(gltf(b.rightHip)), b: (b) => gltf(b.rightAnkle).sub(gltf(b.rightKnee)) },
  { name: "spine", a: (b) => gltf(b.neckPos).sub(gltf(b.hipsPos)), b: (b) => gltf(b.headPos).sub(gltf(b.neckPos)) },
  { name: "shoulderAxis", a: (b) => gltf(b.rightShoulder).sub(gltf(b.leftShoulder)), b: (b) => gltf(b.leftShoulder).sub(gltf(b.rightShoulder)) },
];

const FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;

/** Per-FRAME shape. Note SmplxFrame in smplxConverter.ts is the per-CLIP
 *  wrapper ({label, frames, frameRate}); the per-frame object with `pose` and
 *  `smplx_joints` is SmplxPose. */
function loadFrames(file: string): SmplxPose[] {
  return JSON.parse(fs.readFileSync(file, "utf8")).frames ?? [];
}

async function main() {
  const buggyDir = process.argv[2] ?? path.join(ROOT, "tmp_e7_out_buggy");
  const fixedDir = process.argv[3] ?? path.join(ROOT, "tmp_e7_out");
  if (!fs.existsSync(buggyDir) || !fs.existsSync(fixedDir)) {
    console.error("usage: npx tsx scripts/verify-smplx-e7.ts <buggyDir> <fixedDir>");
    console.error(`  missing: ${!fs.existsSync(buggyDir) ? buggyDir : fixedDir}`);
    process.exit(1);
  }

  const labels = fs
    .readdirSync(buggyDir)
    .filter((f) => f.endsWith(".smplx.json"))
    .map((f) => f.replace(".smplx.json", ""))
    .sort();

  console.log("PHASE 6 — E7 end-to-end impact on the rendered avatar");
  console.log(`buggy: ${buggyDir}`);
  console.log(`fixed: ${fixedDir}\n`);

  let framesCompared = 0;
  let posesWithBody = 0;
  let jointsPopulated = 0;
  let framesSeen = 0;
  const roleMin = new Map<string, number>();
  const handFramesToCompare: [any, any][] = [];
  let worst: { label: string; frame: number; role: string; dot: number } | null = null;

  for (const label of labels) {
    const bf = loadFrames(path.join(buggyDir, `${label}.smplx.json`));
    const ff = loadFrames(path.join(fixedDir, `${label}.smplx.json`));
    if (!bf.length || !ff.length) continue;
    const n = Math.min(bf.length, ff.length);
    framesSeen += n;
    let jn = 0;
    for (let i = 0; i < n; i++) {
      if ((ff[i].smplx_joints ?? []).some((v) => Math.abs(v) > 1e-9)) jn++;
    }
    jointsPopulated += jn;
    const jl = (ff[0].smplx_joints ?? []).length;
    console.log(
      `--- ${label}: ${n} frames | smplx_joints populated ${jn}/${n} ` +
      `(array len ${jl}, expected ${SMPLX_JOINT_COUNT * 3})`
    );

    for (let i = 0; i < n; i++) {
      const pb = smplxPoseToCanonicalPose(bf[i], i / 30);
      const pf = smplxPoseToCanonicalPose(ff[i], i / 30);
      framesCompared++;
      if (!pb || !pf) continue;
      posesWithBody++;
      handFramesToCompare.push([pb, pf]);

      for (const c of DIR_CASES) {
        // Validate the direction is well-formed in the buggy pose, then compare
        // the SAME direction between the two renders. (Comparing a(pf) against
        // b(pb) would mix two different anatomical directions and report a huge
        // fake divergence.)
        const da = c.a(pb.body);
        const db = c.b(pb.body);
        if (da.lengthSq() < 1e-12 || db.lengthSq() < 1e-12) continue;
        const fixedDir = c.a(pf.body);
        if (fixedDir.lengthSq() < 1e-12) continue;
        const dot = fixedDir.normalize().dot(da.normalize());
        if (!Number.isFinite(dot)) continue;
        if (dot < (roleMin.get(c.name) ?? 1)) roleMin.set(c.name, dot);
        if (!worst || dot < worst.dot) worst = { label, frame: i, role: c.name, dot };
      }
    }
  }

  const jointSource = jointsPopulated > 0 ? "smplx_joints (positions)" : "pose (axis-angle)";
  console.log(`\n============ PHASE 6 E7 END-TO-END SUMMARY ============`);
  console.log(`frames compared             : ${framesCompared}`);
  console.log(`frames yielding a body      : ${posesWithBody}`);
  console.log(`frames with joint positions : ${jointsPopulated}/${framesSeen}`);
  console.log(`body source the adapter uses: ${jointSource}`);
  console.log("");

  if (!roleMin.size) {
    console.log("no comparable body directions found");
    process.exitCode = 1;
    return;
  }

  console.log("BODY — per-direction dot between BUGGY-rendered and FIXED-rendered:");
  let minDot = 1;
  for (const [role, dot] of [...roleMin.entries()].sort()) {
    minDot = Math.min(minDot, dot);
    const deg = (Math.acos(Math.min(1, Math.max(-1, dot))) * 180) / Math.PI;
    console.log(`  ${role.padEnd(14)} min dot=${dot.toFixed(6)}  max divergence=${deg.toFixed(3)}deg`);
  }

  // --- HANDS -------------------------------------------------------------
  // The IK layout populates only smplx_joints[0..19]; finger pose comes from
  // the `pose` axis-angle hint at idx 66..71 (joints 22/23) -- exactly the
  // joints E7 corrupted, so the body can be immune while the hands are not.
  console.log("");
  console.log("HANDS — per-finger min dot (driven by the pose[66:72] hint):");
  const fingerMin = new Map<string, number>();
  for (const side of ["leftHand", "rightHand"] as const) {
    for (const f of FINGERS) {
      const k = `${side}.${f}`;
      fingerMin.set(k, 1);
    }
  }
  let handFrames = 0;
  for (let i = 0; i < handFramesToCompare.length; i++) {
    const [pb, pf] = handFramesToCompare[i];
    if (!pb || !pf) continue;
    handFrames++;
    for (const side of ["leftHand", "rightHand"] as const) {
      const hb = pb[side];
      const hf = pf[side];
      for (const f of FINGERS) {
        const db = hb.fingers[f]?.dir;
        const df = hf.fingers[f]?.dir;
        if (!db || !df) continue;
        const vb = new THREE.Vector3(db[0], db[1], db[2]);
        const vf = new THREE.Vector3(df[0], df[1], df[2]);
        if (vb.lengthSq() < 1e-12 || vf.lengthSq() < 1e-12) continue;
        const dot = vb.normalize().dot(vf.normalize());
        const k = `${side}.${f}`;
        if (dot < (fingerMin.get(k) ?? 1)) fingerMin.set(k, dot);
      }
    }
  }
  console.log(`  hand frames compared: ${handFrames}`);
  let minFinger = 1;
  for (const [k, dot] of [...fingerMin.entries()].sort()) {
    minFinger = Math.min(minFinger, dot);
    const deg = (Math.acos(Math.min(1, Math.max(-1, dot))) * 180) / Math.PI;
    console.log(`  ${k.padEnd(20)} min dot=${dot.toFixed(6)}  max divergence=${deg.toFixed(3)}deg`);
  }

  console.log("");
  if (worst) {
    console.log(`worst single frame: ${worst.label} frame ${worst.frame} ${worst.role} dot=${worst.dot.toFixed(6)}`);
  }
  console.log(
    minDot > 0.99999
      ? "BODY VERDICT: E7 did NOT change the rendered body (min dot > 0.99999)."
      : "BODY VERDICT: E7 DID change the rendered body."
  );
  if (jointsPopulated > 0) {
    console.log("  The adapter drives the body from smplx_joints POSITIONS, which are");
    console.log("  bit-identical between buggy and fixed, so the body was immune.");
  }
  console.log(
    minFinger > 0.99999
      ? "HAND VERDICT: E7 did NOT change finger direction."
      : `HAND VERDICT: E7 DID change finger direction (up to ${((Math.acos(minFinger) * 180) / Math.PI).toFixed(1)}deg) —`
  );
  if (minFinger <= 0.99999) {
    console.log("  finger pose is read from the pose[66:72] axis-angle hint, which is");
    console.log("  exactly the data E7 corrupted. Hands of pre-recorded clips were live-buggy.");
  }
  if (minDot <= 0.99999 || minFinger <= 0.99999) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

