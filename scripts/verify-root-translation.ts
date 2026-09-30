/**
 * PHASE 7 — root translation verification.
 *
 * Runs REAL pre-recorded sign clips (the 345-float MediaPipe frames in
 * public/avatar-data) through the entire live pipeline and measures the root
 * translation the avatar would actually receive:
 *
 *   345-float frame -> mediapipeToCanonicalPose -> retarget -> applyBoneRotations
 *                   -> RootTranslationSolver (delta) -> model.position
 *
 * Gates measured:
 *   1. root offset bounding box stays inside the human movement envelope
 *   2. feet never sink below the floor, accounting for the applied root offset
 *   3. no NaN/Inf in any bone matrix or quaternion; every quaternion is unit
 *   4. the offset is a DELTA: the first frame of a sequence is exactly zero
 *
 * Run:  npx tsx scripts/verify-root-translation.ts [label ...]
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import { RootTranslationSolver, DEFAULT_ROOT_TRANSLATION } from "../src/avatar/rootTranslation";
import { LandmarkSmoother } from "../src/utils/landmarkSmoother";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const MODEL_PATH = path.join(ROOT, "public", "models", "Xbot.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FEATURE_DIM = 345;
const DEFAULT_LABELS = ["YES", "HELLO", "THANK_YOU", "MOTHER", "DEAF"];
/** Metres a foot may dip below the avatar's own bind-floor before it counts as
 *  sinking. Covers natural foot roll during extreme signing poses. */
const FOOT_TOLERANCE = 0.01;

function loadFrames(label: string): Float32Array[] {
  const raw = JSON.parse(fs.readFileSync(path.join(DATA_DIR, `${label}.json`), "utf8"));
  const frames: any = raw.frames ?? [];
  const out: Float32Array[] = [];
  if (frames.length && Array.isArray(frames[0])) {
    for (const f of frames) {
      const flat = new Float32Array(FEATURE_DIM);
      for (let i = 0; i < FEATURE_DIM && i < f.length; i++) flat[i] = f[i];
      out.push(flat);
    }
  } else {
    const n = Math.floor(frames.length / FEATURE_DIM);
    for (let i = 0; i < n; i++) {
      out.push(new Float32Array(frames.slice(i * FEATURE_DIM, (i + 1) * FEATURE_DIM)));
    }
  }
  return out;
}

async function main() {
  const labels = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULT_LABELS;
  const available = labels.filter((l) => fs.existsSync(path.join(DATA_DIR, `${l}.json`)));
  if (!available.length) {
    console.error(`no matching labels in ${DATA_DIR}`);
    process.exit(1);
  }

  const scene = new THREE.Scene();
  const raw = fs.readFileSync(MODEL_PATH);
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  const gltf: any = await new Promise((resolve, reject) =>
    new GLTFLoader().parse(ab, "", resolve, reject)
  );
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);
  const sk: ResolvedSkeleton = resolveSkeleton(gltf.scene);

  const bindLowest = (() => {
    let lowest = Infinity;
    let name = "";
    for (const b of sk.bones.values()) {
      const y = b.bone.getWorldPosition(new THREE.Vector3()).y;
      if (y < lowest) { lowest = y; name = b.name; }
    }
    return { y: lowest, name };
  })();

  const C = DEFAULT_ROOT_TRANSLATION;
  console.log("PHASE 7 — ROOT TRANSLATION VERIFICATION");
  console.log(`model      : ${path.basename(MODEL_PATH)} (${sk.bones.size} bones)`);
  console.log(`bind floor : ${bindLowest.name} at y=${bindLowest.y.toFixed(4)} m`);
  console.log(`scale      : ${C.metresPerShoulderWidth} m per shoulder-width unit`);
  console.log(`envelope   : |x|<=${C.maxLateralX}  |z|<=${C.maxLateralZ}  y in [-${C.maxDrop}, +${C.maxLift}]  slew<=${C.maxStep}/frame`);
  console.log("");

  const solver = new RootTranslationSolver();
  const offset = new THREE.Vector3();
  let allPass = true;
  // PHASE 24: the recorded path now applies LandmarkSmoother(3, lowerBodyOnly)
  // before retargeting (see loadLabelDataMediaPipe in dataLoader.ts). The harness
  // must mirror that or it would validate a path the app never uses. App.tsx
  // already did this for LIVE capture; the recorded clips bypassed it entirely,
  // which is why HELLO's foot jitter reached the solver raw.
  const smoother = new LandmarkSmoother(3, /* lowerBodyOnly */ true);
  const grand = { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity, zMin: Infinity, zMax: -Infinity };

  for (const label of available) {
    const frames = loadFrames(label);
    solver.reset();
    smoother.reset();
    const bb = { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity, zMin: Infinity, zMax: -Infinity };
    let nonFinite = 0, badQuatNorm = 0, minFootY = Infinity, firstOffset = NaN, applied = 0;

    for (let i = 0; i < frames.length; i++) {
      const pose = mediapipeToCanonicalPose(smoother.smooth(frames[i]), i / 30);
      if (!pose) continue;
      for (const b of sk.bones.values()) {
        b.bone.quaternion.copy(b.localRot);
        b.bone.position.copy(b.localPos);
      }
      const result = retarget(sk, pose);
      applyBoneRotations(result.boneRotations, sk);
      scene.updateMatrixWorld(true);

      solver.solve(result.hipsWorldPos, offset);
      if (i === 0) firstOffset = offset.length();
      applied++;

      bb.xMin = Math.min(bb.xMin, offset.x); bb.xMax = Math.max(bb.xMax, offset.x);
      bb.yMin = Math.min(bb.yMin, offset.y); bb.yMax = Math.max(bb.yMax, offset.y);
      bb.zMin = Math.min(bb.zMin, offset.z); bb.zMax = Math.max(bb.zMax, offset.z);

      let lowest = Infinity;
      for (const b of sk.bones.values()) {
        const e = b.bone.matrixWorld.elements;
        for (const el of e) if (!Number.isFinite(el)) nonFinite++;
        const q = b.bone.quaternion;
        if (![q.x, q.y, q.z, q.w].every(Number.isFinite)) nonFinite++;
        if (Math.abs(q.length() - 1) > 1e-4) badQuatNorm++;
        const y = b.bone.getWorldPosition(new THREE.Vector3()).y + offset.y;
        if (y < lowest) lowest = y;
      }
      minFootY = Math.min(minFootY, lowest);
    }

    const envOk =
      Math.abs(bb.xMin) <= C.maxLateralX + 1e-6 && Math.abs(bb.xMax) <= C.maxLateralX + 1e-6 &&
      Math.abs(bb.zMin) <= C.maxLateralZ + 1e-6 && Math.abs(bb.zMax) <= C.maxLateralZ + 1e-6 &&
      bb.yMin >= -C.maxDrop - 1e-6 && bb.yMax <= C.maxLift + 1e-6;
    // Footing: measured against the avatar's OWN bind floor, not an absolute
    // y = 0 plane. Xbot's rest pose already puts mixamorigLeftToe_End at
    // y = -0.0029 m, so "y >= 0" was never a property this model has, and an
    // absolute floor would fail the bind pose itself. FOOT_TOLERANCE covers
    // natural foot roll below rest.
    const feetOk = minFootY >= bindLowest.y - FOOT_TOLERANCE;
    const deltaOk = firstOffset === 0;
    const clean = nonFinite === 0 && badQuatNorm === 0;
    const pass = envOk && feetOk && deltaOk && clean;
    if (!pass) allPass = false;

    grand.xMin = Math.min(grand.xMin, bb.xMin); grand.xMax = Math.max(grand.xMax, bb.xMax);
    grand.yMin = Math.min(grand.yMin, bb.yMin); grand.yMax = Math.max(grand.yMax, bb.yMax);
    grand.zMin = Math.min(grand.zMin, bb.zMin); grand.zMax = Math.max(grand.zMax, bb.zMax);

    const f = (n: number) => n.toFixed(4).padStart(8);
    console.log(`${label}  (${applied} frames)  ${pass ? "PASS" : "FAIL"}`);
    console.log(`   root bbox  X [${f(bb.xMin)} ..${f(bb.xMax)}]  Y [${f(bb.yMin)} ..${f(bb.yMax)}]  Z [${f(bb.zMin)} ..${f(bb.zMax)}] m`);
    console.log(`   env=${envOk ? "ok" : "FAIL"}  feet=${feetOk ? "ok" : "FAIL"}(minY=${minFootY.toFixed(4)})  delta0=${deltaOk ? "ok" : "FAIL"}  finite/unit=${clean ? "ok" : "FAIL"}`);
    if (nonFinite) console.log(`   non-finite: ${nonFinite}`);
    if (badQuatNorm) console.log(`   non-unit quats: ${badQuatNorm}`);
    console.log("");
  }

  console.log("============ PHASE 7 ROOT BOUNDING BOX (all clips) ============");
  const g = (n: number) => n.toFixed(4);
  console.log(`X : [${g(grand.xMin)} .. ${g(grand.xMax)}]   span ${g(grand.xMax - grand.xMin)} m`);
  console.log(`Y : [${g(grand.yMin)} .. ${g(grand.yMax)}]   span ${g(grand.yMax - grand.yMin)} m`);
  console.log(`Z : [${g(grand.zMin)} .. ${g(grand.zMax)}]   span ${g(grand.zMax - grand.zMin)} m`);
  const specOk =
    Math.abs(grand.xMin) < 1 && Math.abs(grand.xMax) < 1 &&
    Math.abs(grand.yMin) < 0.5 && Math.abs(grand.yMax) < 0.5 &&
    Math.abs(grand.zMin) < 1 && Math.abs(grand.zMax) < 1;
  console.log(`spec gate |x|<1.0  |y|<0.5  |z|<1.0 : ${specOk ? "PASS" : "FAIL"}`);
  console.log(allPass && specOk ? "\nROOT TRANSLATION VERIFY PASSED" : "\nROOT TRANSLATION VERIFY FAILED");
  if (!allPass || !specOk) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

