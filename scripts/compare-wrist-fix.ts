/**
 * Before/after comparison of the PHASE 44 wrist fix, in the real pipeline.
 *
 * Runs each sampled frame of 3 clips twice:
 *   NEW = body.wrist tracks the detected hand's wristPos (the fix)
 *   OLD = body.wrist tracks the MediaPipe POSE wrist (the bug)
 *
 * Reports the hand positions the avatar would actually render, so the
 * difference is what the user sees.
 *
 * Run: npx tsx scripts/compare-wrist-fix.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { resolveSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import { RootTranslationSolver } from "../src/avatar/rootTranslation";
import { LandmarkSmoother } from "../src/utils/landmarkSmoother";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const GLB = path.join(ROOT, "public", "models", "Avery.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const MEDIAPIPE_FRAME_FLOATS = 345;

function loadClip(label: string): Float32Array[] {
  const p = path.join(DATA_DIR, `${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`);
  const data = JSON.parse(fs.readFileSync(p, "utf8"));
  const frames = data.frames;
  const n = Math.floor(frames.length / MEDIAPIPE_FRAME_FLOATS);
  const out: Float32Array[] = [];
  for (let i = 0; i < n; i++) out.push(new Float32Array(frames.slice(i * MEDIAPIPE_FRAME_FLOATS, (i + 1) * MEDIAPIPE_FRAME_FLOATS)));
  return out;
}

async function loadGlb(): Promise<THREE.Group> {
  const loader = new GLTFLoader();
  const raw = fs.readFileSync(GLB);
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  return new Promise((res, rej) => loader.parse(buffer, "", (g: any) => res(g.scene), rej));
}

function canon(mp: Float32Array, base: number): [number, number, number] {
  return [-mp[base + 2], -mp[base + 1], -mp[base + 0]];
}

async function main() {
  const scene = await loadGlb();
  const sk = resolveSkeleton(scene, { armatureScale: 0.01 });
  const solver = new RootTranslationSolver();
  const offset = new THREE.Vector3();

  const CLIPS = [
    { label: "ADOPT", samples: [0, 19, 39, 59, 78] },
    { label: "HELLO", samples: [0, 52, 104, 156, 207] },
    { label: "A", samples: [0, 23, 47, 70, 93] },
  ];

  let maxDelta = 0;
  let maxDeltaFrame = "";
  for (const { label, samples } of CLIPS) {
    const clip = loadClip(label);
    const smoother = new LandmarkSmoother(3, true, true);
    console.log(`\n=== ${label} ===`);
    console.log(`  frame | NEW Lhand            | NEW Rhand            | OLD Lhand            | OLD Rhand            | hand-move delta (m)`);
    for (const i of samples) {
      const raw = clip[i];
      const buf = smoother.smooth(raw);

      // NEW: the fix (mediapipeToCanonicalPose already applies it)
      const poseNew = mediapipeToCanonicalPose(buf, i / clip.length);

      // OLD: revert the fix by pointing body.wrist back at the POSE wrist
      const poseOld = JSON.parse(JSON.stringify(poseNew)) as typeof poseNew;
      if (!poseOld) continue;
      poseOld.body.leftWrist = canon(raw, 15 * 3);
      poseOld.body.rightWrist = canon(raw, 16 * 3);

      for (const [tag, pose] of [["NEW", poseNew], ["OLD", poseOld]] as const) {
        if (!pose) continue;
        for (const b of sk.bones.values()) {
          b.bone.quaternion.copy(b.localRot);
          b.bone.position.copy(b.localPos);
        }
        const r = retarget(sk, pose);
        applyBoneRotations(r.boneRotations, r.bonePositions, sk);
        scene.updateMatrixWorld(true);
        solver.solve(r.hipsWorldPos, offset);
        scene.position.copy(offset);
        scene.updateMatrixWorld(true);

        const lh = sk.getBone("leftHand")?.bone.getWorldPosition(new THREE.Vector3());
        const rh = sk.getBone("rightHand")?.bone.getWorldPosition(new THREE.Vector3());
        const f = (v?: THREE.Vector3) => v ? `[${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)}]` : "?";
        if (tag === "NEW") {
          (globalThis as any).__newLh = lh;
          (globalThis as any).__newRh = rh;
          console.log(`  ${String(i).padStart(3)} | NEW | ${f(lh)} | ${f(rh)}`);
        } else {
          const nl = (globalThis as any).__newLh as THREE.Vector3;
          const nr = (globalThis as any).__newRh as THREE.Vector3;
          const d = lh && nl ? Math.max(lh.distanceTo(nl), rh ? rh.distanceTo(nr) : 0) : 0;
          if (d > maxDelta) { maxDelta = d; maxDeltaFrame = `${label} f${i}`; }
          console.log(`       | OLD | ${f(lh)} | ${f(rh)} | delta=${d.toFixed(3)} m`);
        }
      }
    }
  }
  console.log(`\nmax hand displacement between OLD and NEW: ${maxDelta.toFixed(3)} m at ${maxDeltaFrame}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
