/**
 * PHASE 8 - ADOPT end-to-end playback validation.
 *
 * The Phase 8 brief describes "the ADOPT sign-language playback/recognition
 * system" as a separate layer. It is not: there is no ADOPT module, service or
 * code path in this repo -- the only matches on disk are DATA files
 * (public/avatar-data-smplx/ADOPT.smplx.json and its backup). ADOPT is one
 * gloss that happens to be in the dataset.
 *
 * So this validates what actually exists: playback of the regenerated SMPL-X
 * clips through the real loader -> adapter -> retargeter chain, with structural
 * checks that would surface a malformed regeneration.
 *
 * Run: npx tsx scripts/verify-adopt-playback.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { smplxPoseToCanonicalPose } from "../src/avatar/smplxAdapter";
import { RootTranslationSolver } from "../src/avatar/rootTranslation";
import type { SmplxPose } from "../src/avatar/smplxConverter";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const SMPLX_DIR = path.join(ROOT, "public", "avatar-data-smplx");
const MODEL_PATH = path.join(ROOT, "public", "models", "Xbot.glb");
const FOOT_TOLERANCE = 0.01;

/** Complex signs: two-handed, facial proximity, large body crossing. */
const TARGETS = [
  { label: "ADOPT", why: "two-handed / body-level compound sign" },
  { label: "UNDERSTAND", why: "two-handed, sign travels across the body" },
  { label: "YESTERDAY", why: "longest clip (75s), largest body excursion" },
];

const f3 = (n: number) => n.toFixed(4);

function loadClip(label: string): SmplxPose[] {
  return JSON.parse(fs.readFileSync(path.join(SMPLX_DIR, `${label}.smplx.json`), "utf8")).frames ?? [];
}

/** Structural validation of a regenerated clip. */
function structuralErrors(frames: SmplxPose[]): string[] {
  const errs: string[] = [];
  if (!frames.length) return ["clip has no frames"];
  const f0 = frames[0];
  if (!Array.isArray(f0.pose) || f0.pose.length !== 72) errs.push(`pose len ${f0.pose?.length} != 72`);
  if (!Array.isArray(f0.smplx_joints) || f0.smplx_joints.length !== 384) {
    errs.push(`smplx_joints len ${f0.smplx_joints?.length} != 384`);
  }
  if (!Array.isArray(f0.betas) || f0.betas.length !== 10) errs.push(`betas len ${f0.betas?.length} != 10`);
  let nonFinite = 0;
  for (const fr of frames) {
    for (const v of [...fr.pose, ...fr.smplx_joints, ...fr.transl, ...fr.global_orient]) {
      if (!Number.isFinite(v)) nonFinite++;
    }
  }
  if (nonFinite) errs.push(`${nonFinite} non-finite floats`);
  return errs;
}

async function main() {
  const available = TARGETS.filter((t) =>
    fs.existsSync(path.join(SMPLX_DIR, `${t.label}.smplx.json`))
  );
  if (!available.length) {
    console.error("none of the target clips exist");
    process.exit(1);
  }

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

  console.log("PHASE 8 - ADOPT / SMPL-X PLAYBACK VALIDATION");
  console.log(`model: ${path.basename(MODEL_PATH)} (${sk.bones.size} bones), bind floor y=${f3(bindFloor)}\n`);

  let allPass = true;
  let allPlaybackOk = true;
  for (const t of available) {
    const frames = loadClip(t.label);
    const errs = structuralErrors(frames);
    const solver = new RootTranslationSolver();
    const offset = new THREE.Vector3();

    let nonFinite = 0;
    let badQuat = 0;
    let minFoot = Infinity;
    let nullPoses = 0;
    let maxSrcHips = 0;
    let framesBelowFloor = 0;
    const wSep: number[] = [];

    for (let i = 0; i < frames.length; i++) {
      const pose = smplxPoseToCanonicalPose(frames[i], i / 30);
      if (!pose) {
        nullPoses++;
        continue;
      }
      for (const b of sk.bones.values()) {
        b.bone.quaternion.copy(b.localRot);
        b.bone.position.copy(b.localPos);
      }
      const r = retarget(sk, pose);
      applyBoneRotations(r.boneRotations, sk);
      scene.updateMatrixWorld(true);
      solver.solve(r.hipsWorldPos, offset);
      gltf.scene.position.copy(offset);
      scene.updateMatrixWorld(true);

      maxSrcHips = Math.max(maxSrcHips, r.hipsWorldPos.length());
      wSep.push(
        Math.hypot(
          pose.body.rightWrist[0] - pose.body.leftWrist[0],
          pose.body.rightWrist[2] - pose.body.leftWrist[2]
        )
      );

      let lowest = Infinity;
      for (const b of sk.bones.values()) {
        for (const e of b.bone.matrixWorld.elements) if (!Number.isFinite(e)) nonFinite++;
        const q = b.bone.quaternion;
        if (![q.x, q.y, q.z, q.w].every(Number.isFinite)) nonFinite++;
        else if (Math.abs(q.length() - 1) > 1e-4) badQuat++;
        lowest = Math.min(lowest, b.bone.getWorldPosition(new THREE.Vector3()).y);
      }
      minFoot = Math.min(minFoot, lowest);
      if (lowest < bindFloor - FOOT_TOLERANCE) framesBelowFloor++;
    }

    const feetOk = minFoot >= bindFloor - FOOT_TOLERANCE;
    // Two independent verdicts. The Phase 8 ADOPT gate is about the app layer
    // loading / parsing / playing back the regenerated clips without structural
    // errors, which footing is not part of. Footing is reported separately
    // because a long clip with large lower-body motion can dip a foot below the
    // bind floor purely through leg ROTATION (the root offset is clamped to
    // y >= 0, so it can only ever lift). That is a real Phase 9 candidate and
    // is deliberately NOT clamped away here.
    const playbackOk = errs.length === 0 && nonFinite === 0 && badQuat === 0 && nullPoses === 0;
    const pass = playbackOk && feetOk;
    if (!playbackOk) allPlaybackOk = false;
    if (!pass) allPass = false;
    const wRange = wSep.length ? Math.max(...wSep) - Math.min(...wSep) : 0;

    console.log(`${t.label}  (${t.why})`);
    console.log(`   frames played     : ${frames.length - nullPoses}/${frames.length} (null poses ${nullPoses})`);
    console.log(`   structural errors : ${errs.length ? errs.join("; ") : "none"}`);
    console.log(`   NaN/Inf           : ${nonFinite}`);
    console.log(`   non-unit quats    : ${badQuat}`);
    console.log(`   min foot y        : ${f3(minFoot)} (bind floor ${f3(bindFloor)}) ${feetOk ? "ok" : "SUNK"}`);
    console.log(`   frames below floor: ${framesBelowFloor}/${frames.length - nullPoses} ` +
      `(${(100 * framesBelowFloor / Math.max(1, frames.length - nullPoses)).toFixed(2)}% of frames)`);
    console.log(`   |source hips|     : ${f3(maxSrcHips)} shoulder-width units`);
    console.log(`   wrist separation  : ${f3(wRange)} shoulder-width units (motion extent)`);
    console.log(`   playback verdict  : ${playbackOk ? "PASS" : "FAIL"}  (load/parse/play, no structural errors)`);
    console.log(`   footing verdict   : ${feetOk ? "PASS" : "FAIL"}  (foot below bind floor on ` +
      `${(100 * framesBelowFloor / Math.max(1, frames.length - nullPoses)).toFixed(1)}% of frames)`);
    console.log("");
  }

  console.log(allPass
    ? "ADOPT / SMPL-X PLAYBACK PASSED (playback + footing)"
    : allPlaybackOk
      ? "ADOPT PLAYBACK PASSED (load/parse/play, no structural errors)."
      + "\nFOOTING ISSUE on long clips -> Phase 9 candidate, NOT clamped away."
      : "ADOPT / SMPL-X PLAYBACK FAILED (structural or numerical error)");
  if (!allPlaybackOk) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

