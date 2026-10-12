/**
 * ASCII hand visualizer — gives the agent "eyes" via text.
 *
 * I cannot view images, so this renders each hand as an ASCII
 * grid in two orthographic views:
 *   FRONT: canonical X (right) vs Y (up)     — what the user sees
 *   TOP:   canonical X (right) vs Z (depth)  — palm-up/down check
 *
 * The recordings are already shoulder-normalized (shoulder midpoint
 * at origin, shoulder width = 1 unit), so landmarks read straight
 * from the buffer are directly comparable to the rendered skeleton
 * ONCE the rendered joints are expressed in the same frame:
 *   rendered_norm = (gltfToCanonical(worldPos) - renderedShoulderMid)
 *                   / renderedShoulderDist
 *
 * Legend:
 *   'o' = MediaPipe landmark (ground truth)
 *   '#' = rendered Xbot bone joint (what the avatar shows)
 *   '*' = both in the same cell (good overlap)
 *
 * Plus numeric arm-chain errors (elbow / wrist / fingertip, in
 * shoulder-width units) and per-finger bend angles.
 *
 * Run: npx tsx scripts/visualize-fingers.ts [recordings...]
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
const POSE_OFFSET = 0;
const HAND = { WRIST: 0, INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
  THUMB_MCP: 1, THUMB_PIP: 2, THUMB_DIP: 3, THUMB_TIP: 4 };
const POSE = { LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16 };

type FingerKey = "thumb" | "index" | "middle" | "ring" | "pinky";
type Side = "left" | "right";
const FINGERS: { key: FingerKey; idxs: number[] }[] = [
  { key: "index",  idxs: [HAND.INDEX_MCP, HAND.INDEX_PIP, HAND.INDEX_DIP, HAND.INDEX_TIP] },
  { key: "middle", idxs: [HAND.MIDDLE_MCP, HAND.MIDDLE_PIP, HAND.MIDDLE_DIP, HAND.MIDDLE_TIP] },
  { key: "ring",   idxs: [HAND.RING_MCP, HAND.RING_PIP, HAND.RING_DIP, HAND.RING_TIP] },
  { key: "pinky",  idxs: [HAND.PINKY_MCP, HAND.PINKY_PIP, HAND.PINKY_DIP, HAND.PINKY_TIP] },
  { key: "thumb",  idxs: [HAND.THUMB_MCP, HAND.THUMB_PIP, HAND.THUMB_DIP, HAND.THUMB_TIP] },
];

function readHandCanon(data: Float32Array, groupOffset: number): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < 21; i++) {
    const b = groupOffset + i * 3;
    out.push(mediapipeToCanonical([data[b], data[b + 1], data[b + 2]]) as Vec3);
  }
  return out;
}

function readPoseCanon(data: Float32Array, index: number): Vec3 {
  const b = POSE_OFFSET + index * 3;
  return mediapipeToCanonical([data[b], data[b + 1], data[b + 2]]) as Vec3;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l < 1e-10 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
};
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const bendDeg = (mcp: Vec3, pip: Vec3, dip: Vec3): number => {
  const v1 = norm(sub(pip, mcp));
  const v2 = norm(sub(dip, pip));
  return (Math.acos(Math.min(1, Math.max(-1, -dot(v1, v2)))) * 180) / Math.PI;
};

const GW = 44;
const GH = 20;

function renderGrid(lms: Vec3[], rendered: (Vec3 | null)[], view: "front" | "top", title: string): string {
  const proj = (p: Vec3): [number, number] =>
    view === "front" ? [p[0], p[1]] : [p[0], p[2]];

  const pts: [number, number][] = [];
  for (const p of lms) pts.push(proj(p));
  for (const p of rendered) if (p) pts.push(proj(p));

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const padX = Math.max((maxX - minX) * 0.12, 1e-6);
  const padY = Math.max((maxY - minY) * 0.12, 1e-6);
  minX -= padX; maxX += padX; minY -= padY; maxY += padY;
  const s = Math.min((GW - 1) / (maxX - minX), (GH - 1) / (maxY - minY));
  const col = (x: number) => Math.round((x - minX) * s);
  const row = (y: number) => Math.round((maxY - y) * s);

  const lmSet = new Set<string>();
  for (const p of lms) lmSet.add(`${col(proj(p)[0])},${row(proj(p)[1])}`);
  const rSet = new Set<string>();
  for (const p of rendered) {
    if (!p) continue;
    rSet.add(`${col(proj(p)[0])},${row(proj(p)[1])}`);
  }

  const lines: string[] = [];
  lines.push(`  ${title}`);
  lines.push(`  +${"-".repeat(GW)}+`);
  for (let r = 0; r < GH; r++) {
    let line = "  |";
    for (let c = 0; c < GW; c++) {
      const k = `${c},${r}`;
      const inL = lmSet.has(k);
      const inR = rSet.has(k);
      line += inL && inR ? "*" : inL ? "o" : inR ? "#" : " ";
    }
    lines.push(line + "|");
  }
  lines.push(`  +${"-".repeat(GW)}+`);
  return lines.join("\n");
}

async function main() {
  const recordings =
    process.argv.slice(2).length > 0
      ? process.argv.slice(2)
      : ["FOUR", "FIVE", "THREE", "THANK_YOU", "KNOW"];

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

  const resetPose = () => {
    for (const b of sk.bones.values()) {
      b.bone.quaternion.copy(b.localRot);
      b.bone.position.copy(b.localPos);
    }
  };

  const worldCanon = (bone?: THREE.Object3D): Vec3 | null => {
    if (!bone) return null;
    const wp = bone.getWorldPosition(new THREE.Vector3());
    return gltfToCanonical([wp.x, wp.y, wp.z]) as Vec3;
  };

  // Rendered arm chain in shoulder-normalized canonical units.
  // shoulder landmark 11/12 ~ upperArm origin, elbow 13/14 ~ forearm
  // origin, wrist 15/16 ~ hand origin.
  const renderedFrame = (side: Side) => {
    const arm = sk.getBone((side === "left" ? "leftUpperArm" : "rightUpperArm") as any);
    const fore = sk.getBone((side === "left" ? "leftForearm" : "rightForearm") as any);
    const hand = sk.getBone((side === "left" ? "leftHand" : "rightHand") as any);
    return {
      shoulder: worldCanon(arm?.bone),
      elbow: worldCanon(fore?.bone),
      wrist: worldCanon(hand?.bone),
    };
  };

  // joints for one side's hand in a caller-supplied normalized frame
  const handJoints = (side: Side, toNorm: (p: Vec3) => Vec3): (Vec3 | null)[] => {
    const out: (Vec3 | null)[] = new Array(21).fill(null);
    const w = renderedFrame(side).wrist;
    if (w) out[HAND.WRIST] = toNorm(w);
    for (const f of FINGERS) {
      const chain = names[side][f.key] ?? [];
      for (let k = 0; k < Math.min(4, chain.length); k++) {
        const wp = sk.bones.get(chain[k])!.bone.getWorldPosition(new THREE.Vector3());
        out[f.idxs[k]] = toNorm(gltfToCanonical([wp.x, wp.y, wp.z]) as Vec3);
      }
    }
    return out;
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
    if (frameCount === 0) {
      console.log(`!! ${rec}: no frames`);
      continue;
    }

    const good: number[] = [];
    for (let fi = 0; fi < frameCount; fi++) {
      const frame = flat.slice(fi * FRAME_FLOATS, (fi + 1) * FRAME_FLOATS);
      const pose = mediapipeToCanonicalPose(frame, fi / 24);
      if (pose && lastHandAssignment && (lastHandAssignment.left !== null || lastHandAssignment.right !== null)) {
        good.push(fi);
      }
    }

    console.log(`\n${"=".repeat(78)}`);
    console.log(`${rec}  (${frameCount} frames, ${good.length} with hands)`);
    console.log(`${"=".repeat(78)}`);
    if (good.length === 0) continue;

    const samples: number[] = [];
    const nS = Math.min(3, good.length);
    for (let k = 0; k < nS; k++) {
      samples.push(good[Math.floor((k * (good.length - 1)) / Math.max(1, nS - 1))]);
    }

    for (const fi of samples) {
      const frame = flat.slice(fi * FRAME_FLOATS, (fi + 1) * FRAME_FLOATS);
      const pose = mediapipeToCanonicalPose(frame, fi / 24);
      if (!pose || !lastHandAssignment) continue;
      const assign = lastHandAssignment;

      resetPose();
      const result = retarget(sk, pose);
      applyBoneRotations(result.boneRotations, sk);
      scene.updateMatrixWorld(true);

      // rendered shoulder frame (both shoulders, from the skeleton)
      const ls = renderedFrame("left").shoulder;
      const rs = renderedFrame("right").shoulder;
      if (!ls || !rs) {
        console.log("  (skeleton shoulder bones not found)");
        continue;
      }
      const sMid = scale(add(ls, rs), 0.5);
      const sDist = Math.max(dist(ls, rs), 1e-6);
      const toNorm = (p: Vec3): Vec3 => scale(sub(p, sMid), 1 / sDist);

      // landmark shoulder mid (should be ~origin)
      const lmSMid = scale(add(readPoseCanon(frame, POSE.LEFT_SHOULDER), readPoseCanon(frame, POSE.RIGHT_SHOULDER)), 0.5);

      for (const side of ["left", "right"] as Side[]) {
        const groupOffset = assign[side];
        if (groupOffset === null) continue;
        const lms = readHandCanon(frame, groupOffset);
        const rj = handJoints(side, toNorm);

        // numeric arm-chain errors (shoulder-width units)
        const lmElbow = readPoseCanon(frame, side === "left" ? POSE.LEFT_ELBOW : POSE.RIGHT_ELBOW);
        const lmWrist = readPoseCanon(frame, side === "left" ? POSE.LEFT_WRIST : POSE.RIGHT_WRIST);
        const rElbow = renderedFrame(side).elbow;
        const rWrist = renderedFrame(side).wrist;
        const elbowErr = rElbow ? dist(toNorm(rElbow), lmElbow) : NaN;
        const wristErr = rWrist ? dist(toNorm(rWrist), lmWrist) : NaN;
        const lmTip = lms[HAND.MIDDLE_TIP];
        const rTip = rj[HAND.MIDDLE_TIP];
        const tipErr = rTip ? dist(rTip, lmTip) : NaN;

        console.log(`\n-- frame ${fi} / ${frameCount}  ${side.toUpperCase()} hand`);
        console.log(`   shoulder-mid offset (landmark vs rendered origin): ${fmt(lmSMid)} vs (${(sMid[0]).toFixed(3)}, ${(sMid[1]).toFixed(3)}, ${(sMid[2]).toFixed(3)})`);
        console.log(`   arm-chain error (shoulder widths): elbow ${elbowErr.toFixed(3)}   wrist ${wristErr.toFixed(3)}   middle-tip ${tipErr.toFixed(3)}`);

        console.log(renderGrid(lms, rj, "front", "FRONT (X right, Y up)   o=landmark  #=rendered  *=both"));
        console.log(renderGrid(lms, rj, "top", "TOP (X right, Z depth)"));

        console.log("  finger    landmark bend   rendered bend   (deg; 180=straight, 0=folded)");
        for (const f of FINGERS) {
          const lmB = bendDeg(lms[f.idxs[0]], lms[f.idxs[1]], lms[f.idxs[2]]);
          const c1 = rj[f.idxs[0]];
          const c2 = rj[f.idxs[1]];
          const c3 = rj[f.idxs[2]];
          if (!c1 || !c2 || !c3) continue;
          const rB = bendDeg(c1, c2, c3);
          const flag = Math.abs(lmB - rB) > 40 ? "  <-- MISMATCH" : "";
          console.log(
            `  ${f.key.padEnd(8)} ${lmB.toFixed(1).padStart(8)}       ${rB.toFixed(1).padStart(8)}${flag}`
          );
        }
      }
    }
  }
}

const fmt = (p: Vec3) => `(${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)})`;

main().catch((e) => { console.error(e); process.exit(1); });
