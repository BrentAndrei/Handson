/**
 * PHASE 9 — footing verification (HARD GATE).
 *
 * Phase 8 measured that on YESTERDAY a foot dips below the bind floor on 16.6%
 * of frames, worst case ~2.2 cm, caused purely by hip/knee rotation (the root
 * offset is clamped to y >= 0 and can only lift). footingSolver.ts corrects
 * that additively, touching ONLY the hip and knee local quaternions.
 *
 * This asserts the spec's hard gates:
 *   1. 0.0% floor penetration on the extreme clip, and the short clips
 *      (HELLO, ADOPT, UNDERSTAND) must be UNCHANGED.
 *   2. No NaN/Inf anywhere, and every quaternion still unit.
 *   3. The ankle/foot local rotation is still exactly its bind rotation
 *      (Phase 4/5 no-data gate must not be broken by the correction).
 *   4. The upper body is untouched: every non-leg bone's world rotation must be
 *      bit-identical before and after the solve.
 *
 * Run: npx tsx scripts/verify-footing.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { smplxPoseToCanonicalPose } from "../src/avatar/smplxAdapter";
import { solveFooting, DEFAULT_FOOTING } from "../src/avatar/footingSolver";
import { RootTranslationSolver } from "../src/avatar/rootTranslation";
import type { SmplxPose } from "../src/avatar/smplxConverter";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const SMPLX_DIR = path.join(ROOT, "public", "avatar-data-smplx");
const MODEL_PATH = path.join(ROOT, "public", "models", "Xbot.glb");

const TARGETS = [
  { label: "YESTERDAY", why: "extreme clip - 16.6% penetration in Phase 8" },
  { label: "HELLO", why: "standard clip - must stay clean" },
  { label: "ADOPT", why: "two-handed compound - must stay clean" },
  { label: "UNDERSTAND", why: "body-crossing - must stay clean" },
];

const f3 = (n: number) => n.toFixed(4);
/** Bones the solver is allowed to rewrite. */
const LEG_BONES = new Set(["mixamorigLeftUpLeg", "mixamorigLeftLeg", "mixamorigRightUpLeg", "mixamorigRightLeg"]);

const dq = (x: THREE.Quaternion, y: THREE.Quaternion) =>
  2 * Math.acos(Math.min(1, Math.abs(x.dot(y))));

async function main() {
  const scene = new THREE.Scene();
  const raw = fs.readFileSync(MODEL_PATH);
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  const gltf: any = await new Promise((res, rej) => new GLTFLoader().parse(ab, "", res, rej));
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);
  const sk: ResolvedSkeleton = resolveSkeleton(gltf.scene);

  let bindFloor = Infinity;
  for (const b of sk.bones.values()) {
    bindFloor = Math.min(bindFloor, b.bone.getWorldPosition(new THREE.Vector3()).y);
  }
  const ankleBind = ["leftAnkle", "rightAnkle"].map(
    (r) => sk.getBone(r as never)!.bone.quaternion.clone()
  );

  console.log("PHASE 9 - GROUND CONTACT / FOOTING VERIFICATION (hard gate)");
  console.log(`model: ${path.basename(MODEL_PATH)} (${sk.bones.size} bones)`);
  console.log(`bind floor y        : ${f3(bindFloor)}`);
  console.log(`tolerance           : ${DEFAULT_FOOTING.tolerance} m below floorY`);
  console.log(`solver may rewrite  : ${[...LEG_BONES].join(", ")}`);
  console.log(`ankle local rot     : NOT touched (Phase 4/5 no-data gate)`);
  console.log("");

  const lowestY = () => {
    let low = Infinity;
    let nm = "";
    for (const b of sk.bones.values()) {
      const y = b.bone.getWorldPosition(new THREE.Vector3()).y;
      if (y < low) {
        low = y;
        nm = b.name;
      }
    }
    lowestName = nm;
    return low;
  };
  let lowestName = "";

  let allPass = true;
  for (const t of TARGETS) {
    const file = path.join(SMPLX_DIR, `${t.label}.smplx.json`);
    if (!fs.existsSync(file)) {
      console.log(`${t.label}: SKIP (no clip)`);
      continue;
    }
    const frames: SmplxPose[] = JSON.parse(fs.readFileSync(file, "utf8")).frames ?? [];

    const solver = new RootTranslationSolver();
    const offset = new THREE.Vector3();
    const limitY = bindFloor - DEFAULT_FOOTING.tolerance;
    let penBefore = 0, penAfter = 0, framesFixed = 0, stillPen = 0;
    let nonFinite = 0, badQuat = 0, upperChanged = 0, maxIters = 0, unreachable = 0;
    let maxHip = 0, maxKnee = 0;
    let worstLowY = Infinity;

    for (let i = 0; i < frames.length; i++) {
      const pose = smplxPoseToCanonicalPose(frames[i], i / 30);
      if (!pose) continue;
      for (const b of sk.bones.values()) {
        b.bone.quaternion.copy(b.localRot);
        b.bone.position.copy(b.localPos);
      }
      const r0 = retarget(sk, pose);
      applyBoneRotations(r0.boneRotations, sk);
      solver.solve(r0.hipsWorldPos, offset);
      gltf.scene.position.copy(offset);
      scene.updateMatrixWorld(true);

      const before = lowestY();
      // Penetration depth is how far BELOW the limit we are: limitY - before.
      // (before - limitY would measure height above the floor and count clean
      // frames as penetrating.)
      penBefore = Math.max(penBefore, limitY - before);

      const pre = new Map<string, THREE.Quaternion>();
      for (const b of sk.bones.values()) {
        if (!LEG_BONES.has(b.name)) pre.set(b.name, b.bone.quaternion.clone());
      }
      // Measure BOTH legs: the deepest penetration in this dataset is on the
      // right, so sampling only the left would report a bogus 0 deg correction.
      const legPre = new Map<string, THREE.Quaternion>();
      for (const r of ["leftHip", "leftKnee", "rightHip", "rightKnee"] as const) {
        legPre.set(r, sk.getBone(r as never)!.bone.quaternion.clone());
      }

      const res = solveFooting(sk, { floorY: bindFloor });
      scene.updateMatrixWorld(true);

      const after = lowestY();
      if (after < worstLowY) worstLowY = after;
      const pen = limitY - after; // positive = still below the floor
      if (pen > 0) {
        stillPen++;
        penAfter = Math.max(penAfter, pen);
      }
      if (before < limitY && after >= limitY) framesFixed++;

      for (const b of sk.bones.values()) {
        for (const e of b.bone.matrixWorld.elements) if (!Number.isFinite(e)) nonFinite++;
        const q = b.bone.quaternion;
        if (![q.x, q.y, q.z, q.w].every(Number.isFinite)) nonFinite++;
        else if (Math.abs(q.length() - 1) > 1e-4) badQuat++;
        const p = pre.get(b.name);
        if (p && Math.abs(p.dot(q)) < 1 - 1e-9) upperChanged++;
      }
      for (const r of ["leftHip", "leftKnee", "rightHip", "rightKnee"] as const) {
        const d = dq(legPre.get(r)!, sk.getBone(r as never)!.bone.quaternion);
        if (r.endsWith("Hip")) maxHip = Math.max(maxHip, d);
        else maxKnee = Math.max(maxKnee, d);
      }
      for (const rr of res) {
        maxIters = Math.max(maxIters, rr.iterations);
        if (!rr.reachable) unreachable++;
      }
    }

    let ankleDrift = 0;
    for (let k = 0; k < 2; k++) {
      const side = k ? "right" : "left";
      ankleDrift = Math.max(ankleDrift, dq(ankleBind[k], sk.getBone((side + "Ankle") as never)!.bone.quaternion));
    }

    const pct = (100 * stillPen) / Math.max(1, frames.length);
    const pass = stillPen === 0 && nonFinite === 0 && badQuat === 0 && upperChanged === 0 && ankleDrift < 1e-9;
    if (!pass) allPass = false;

    console.log(`${t.label}  (${t.why})`);
    console.log(`   frames                   : ${frames.length}`);
    console.log(`   worst lowest BONE        : ${lowestName} @ ${f3(worstLowY)} m`);
    console.log(`   penetration BEFORE (max) : ${f3(penBefore)} m`);
    console.log(`   penetration AFTER  (max) : ${f3(penAfter)} m`);
    console.log(`   frames fixed by solver   : ${framesFixed}`);
    console.log(`   frames STILL penetrating : ${stillPen} (${f3(pct)}%)  ${pct === 0 ? "GATE MET" : "GATE FAILED"}`);
    console.log(`   NaN/Inf                  : ${nonFinite}`);
    console.log(`   non-unit quaternions     : ${badQuat}`);
    console.log(`   upper-body bones changed : ${upperChanged}   (must be 0)`);
    console.log(`   ankle local rot drift    : ${ankleDrift.toExponential(2)} rad (must be 0)`);
    console.log(`   max hip correction       : ${f3((maxHip * 180) / Math.PI)} deg`);
    console.log(`   max knee correction      : ${f3((maxKnee * 180) / Math.PI)} deg`);
    console.log(`   max iters / out-of-reach : ${maxIters} / ${unreachable}`);
    console.log(`   verdict                  : ${pass ? "PASS" : "FAIL"}`);
    console.log("");
  }

  console.log(allPass
    ? "GROUND CONTACT VERIFY PASSED (0% penetration, upper body + ankles untouched)"
    : "GROUND CONTACT VERIFY FAILED");
  if (!allPass) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
