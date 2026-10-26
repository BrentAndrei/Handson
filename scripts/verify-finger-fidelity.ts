/**
 * Ground-truth finger fidelity diagnostic.
 *
 * The verify-retarget suite checks flexion MAGNITUDE but never the
 * DIRECTION the finger curls (toward vs away from the palm), nor
 * whether the rendered fingertip actually points where the MediaPipe
 * landmarks say it points. This script does both against real
 * recordings from public/avatar-data:
 *
 *   for each frame:
 *     landmarks -> mediapipeToCanonicalPose -> retarget -> bones
 *     then compares, per finger, in canonical world space:
 *       1. tip direction  : rendered (Index1 -> Index4 world dir)
 *                           vs landmark (MCP -> TIP dir)
 *       2. curl direction : rendered bend (2nd difference of the
 *                           rendered chain) vs landmark bend
 *                           (2nd difference of MCP/PIP/DIP)
 *          dot < 0  -> the avatar curls the finger the WRONG WAY
 *
 * Run with: npx tsx scripts/verify-finger-fidelity.ts [recordings...]
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, getFingerBoneNames, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose, lastHandAssignment } from "../src/avatar/mediapipeAdapter";
import { gltfToCanonical, mediapipeToCanonical } from "../src/avatar/coordinateSystem";
import type { Vec3 } from "../src/avatar/canonicalPose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const MODEL_PATH = path.join(ROOT, "public", "models", "Xbot.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FRAME_FLOATS = 345;
const HAND = { WRIST: 0, INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
  THUMB_MCP: 1, THUMB_PIP: 2, THUMB_DIP: 3, THUMB_TIP: 4 };

type FingerKey = "thumb" | "index" | "middle" | "ring" | "pinky";
const FINGERS: { key: FingerKey; mcp: number; pip: number; dip: number; tip: number }[] = [
  { key: "index", mcp: HAND.INDEX_MCP, pip: HAND.INDEX_PIP, dip: HAND.INDEX_DIP, tip: HAND.INDEX_TIP },
  { key: "middle", mcp: HAND.MIDDLE_MCP, pip: HAND.MIDDLE_PIP, dip: HAND.MIDDLE_DIP, tip: HAND.MIDDLE_TIP },
  { key: "ring", mcp: HAND.RING_MCP, pip: HAND.RING_PIP, dip: HAND.RING_DIP, tip: HAND.RING_TIP },
  { key: "pinky", mcp: HAND.PINKY_MCP, pip: HAND.PINKY_PIP, dip: HAND.PINKY_DIP, tip: HAND.PINKY_TIP },
  { key: "thumb", mcp: HAND.THUMB_MCP, pip: HAND.THUMB_PIP, dip: HAND.THUMB_DIP, tip: HAND.THUMB_TIP },
];

/** Read the 21 hand landmarks of one group as canonical Vec3s. */
function readHandCanon(data: Float32Array, groupOffset: number): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < 21; i++) {
    const b = groupOffset + i * 3;
    const raw: Vec3 = [data[b], data[b + 1], data[b + 2]];
    out.push(mediapipeToCanonical(raw) as Vec3);
  }
  return out;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l < 1e-10 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
};
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const ang = (a: Vec3, b: Vec3) =>
  (Math.acos(Math.min(1, Math.max(-1, dot(norm(a), norm(b))))) * 180) / Math.PI;

async function main() {
  const recordings =
    process.argv.slice(2).length > 0
      ? process.argv.slice(2)
      : ["FOUR", "FIVE", "THREE", "KNOW", "THANK_YOU", "I_LOVE_YOU"];

  const scene = new THREE.Scene();
  const raw = fs.readFileSync(MODEL_PATH);
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  const loader = new GLTFLoader();
  const gltf: any = await new Promise((resolve, reject) => {
    loader.parse(buffer, "", resolve, reject);
  });
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);
  const sk: ResolvedSkeleton = resolveSkeleton(gltf.scene);

  const names = {
    left: getFingerBoneNames(sk, "left"),
    right: getFingerBoneNames(sk, "right"),
  };

  // Per-finger accumulators: [side][fingerKey]
  type Acc = { frames: number; tipAngleSum: number; tipAngleMax: number; curlFlip: number; curlDotSum: number; curlPresent: number };
  const accs: Record<string, Record<FingerKey, Acc>> = {
    left: {} as any, right: {} as any,
  };
  for (const side of ["left", "right"] as const) {
    for (const f of FINGERS) {
      accs[side][f.key] = { frames: 0, tipAngleSum: 0, tipAngleMax: 0, curlFlip: 0, curlDotSum: 0, curlPresent: 0 };
    }
  }

  let totalFrames = 0;
  let poseNull = 0;

  for (const rec of recordings) {
    const file = path.join(DATA_DIR, rec.endsWith(".json") ? rec : `${rec}.json`);
    if (!fs.existsSync(file)) {
      console.log(`!! ${rec}: not found`);
      continue;
    }
    const j = JSON.parse(fs.readFileSync(file, "utf8")) as { frames?: number[] };
    const flat = new Float32Array(j.frames ?? []);
    const frameCount = Math.floor(flat.length / FRAME_FLOATS);
    if (frameCount === 0) {
      console.log(`!! ${rec}: no frames`);
      continue;
    }

    let recFrames = 0;
    for (let fi = 0; fi < frameCount; fi += 2) {
      const frame = flat.slice(fi * FRAME_FLOATS, (fi + 1) * FRAME_FLOATS);
      const pose = mediapipeToCanonicalPose(frame, fi / 24);
      totalFrames++;
      if (!pose) { poseNull++; continue; }
      const assign = lastHandAssignment;
      if (!assign) continue;

      // Retarget this pose onto the skeleton.
      for (const b of sk.bones.values()) {
        b.bone.quaternion.copy(b.localRot);
        b.bone.position.copy(b.localPos);
      }
      const result = retarget(sk, pose);
      applyBoneRotations(result.boneRotations, result.bonePositions, sk);
      scene.updateMatrixWorld(true);
      recFrames++;

      for (const side of ["left", "right"] as const) {
        const groupOffset = side === "left" ? assign.left : assign.right;
        if (groupOffset === null) continue;
        const lms = readHandCanon(frame, groupOffset);
        const sideNames = names[side];

        for (const f of FINGERS) {
          const a = accs[side][f.key];

          // --- landmark-implied directions (canonical) ---
          const lmTipDir = norm(sub(lms[f.tip], lms[f.mcp]));
          // landmark curl: 2nd difference (dip - pip) - (pip - mcp)
          const lmBend = sub(sub(lms[f.dip], lms[f.pip]), sub(lms[f.pip], lms[f.mcp]));
          const lmBendLen = Math.hypot(lmBend[0], lmBend[1], lmBend[2]);

          // --- rendered chain: world positions of the 4 phalanx bones ---
          const chain = sideNames[f.key] ?? [];
          if (chain.length < 4) continue;
          const p1 = sk.bones.get(chain[0])!.bone.getWorldPosition(new THREE.Vector3());
          const p2 = sk.bones.get(chain[1])!.bone.getWorldPosition(new THREE.Vector3());
          const p3 = sk.bones.get(chain[2])!.bone.getWorldPosition(new THREE.Vector3());
          const p4 = sk.bones.get(chain[3])!.bone.getWorldPosition(new THREE.Vector3());
          // GLTF world -> canonical
          const c1 = gltfToCanonical([p1.x, p1.y, p1.z]) as Vec3;
          const c2 = gltfToCanonical([p2.x, p2.y, p2.z]) as Vec3;
          const c3 = gltfToCanonical([p3.x, p3.y, p3.z]) as Vec3;
          const c4 = gltfToCanonical([p4.x, p4.y, p4.z]) as Vec3;

          const rTipDir = norm(sub(c4, c1));
          const rBend = sub(sub(c3, c2), sub(c2, c1));

          // tip direction fidelity
          const tipAng = ang(rTipDir, lmTipDir);
          a.frames++;
          a.tipAngleSum += tipAng;
          a.tipAngleMax = Math.max(a.tipAngleMax, tipAng);

          // curl direction fidelity (only when the landmark bend is
          // substantial, i.e. the finger is genuinely curved)
          if (lmBendLen > 0.05) {
            const d = dot(norm(rBend), norm(lmBend));
            a.curlPresent++;
            a.curlDotSum += d;
            if (d < 0) a.curlFlip++;
          }
        }
      }
    }
    console.log(`${rec}: ${recFrames} frames measured`);
  }

  console.log(`\ntotal frames: ${totalFrames} (pose=null: ${poseNull})`);
  console.log("\n=== FINGER FIDELITY (rendered vs MediaPipe landmarks, canonical world space) ===");
  console.log("side   finger  n     tipAngle mean/max deg   curlFlip %   curlDot mean");
  let worstMean = 0;
  let worstFlip = 0;
  for (const side of ["left", "right"] as const) {
    for (const f of FINGERS) {
      const a = accs[side][f.key];
      if (a.frames === 0) continue;
      const mean = a.tipAngleSum / a.frames;
      const flipPct = a.curlPresent ? (100 * a.curlFlip) / a.curlPresent : 0;
      const curlDot = a.curlPresent ? a.curlDotSum / a.curlPresent : NaN;
      worstMean = Math.max(worstMean, mean);
      worstFlip = Math.max(worstFlip, flipPct);
      console.log(
        side.padEnd(6) + f.key.padEnd(7) +
        String(a.frames).padEnd(6) +
        mean.toFixed(2).padStart(6) + " / " + a.tipAngleMax.toFixed(1).padStart(6) +
        "     " + flipPct.toFixed(1).padStart(6) + "%" +
        "      " + (Number.isFinite(curlDot) ? curlDot.toFixed(3) : "n/a")
      );
    }
  }
  console.log(`\nworst mean tip-angle error: ${worstMean.toFixed(2)} deg`);
  console.log(`worst curl-direction flip rate: ${worstFlip.toFixed(1)}%`);
}

main().catch((e) => { console.error(e); process.exit(1); });
