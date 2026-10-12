/**
 * Isolate whether the bend-vote assignment override mirrors hands.
 *
 * Compares the rendered hand position (rebased to the rendered
 * shoulder frame) against BOTH hand-slot landmarks and both pose
 * wrists, for each side, and reports which source the rendered
 * hand actually matches.
 *
 * A correct pipeline: rendered LEFT hand lands on the landmark
 * group assigned to left. If it lands on the other slot's
 * landmarks, the assignment is swapping hands.
 *
 * Run: npx tsx scripts/diagnose-mirror.ts A D O P T
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose, lastHandAssignment } from "../src/avatar/mediapipeAdapter";
import { gltfToCanonical, mediapipeToCanonical } from "../src/avatar/coordinateSystem";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const MODEL = path.join(ROOT, "public", "models", "Xbot.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FF = 345;
const LEFT_SLOT = 99;
const RIGHT_SLOT = 162;

async function main() {
  const recs = process.argv.slice(2).length ? process.argv.slice(2) : ["A", "D", "O", "P", "T"];

  const scene = new THREE.Scene();
  const raw = fs.readFileSync(MODEL);
  const gltf: any = await new Promise((res, rej) =>
    new GLTFLoader().parse(
      raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
      "", res, rej
    )
  );
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);
  const sk: ResolvedSkeleton = resolveSkeleton(gltf.scene);

  const reset = () => {
    for (const b of sk.bones.values()) {
      b.bone.quaternion.copy(b.localRot);
      b.bone.position.copy(b.localPos);
    }
  };
  const wc = (bone?: THREE.Object3D) => {
    if (!bone) return null;
    const p = bone.getWorldPosition(new THREE.Vector3());
    return gltfToCanonical([p.x, p.y, p.z]) as [number, number, number];
  };
  const dist = (a: [number, number, number], b: [number, number, number]) =>
    Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const handWristLm = (data: Float32Array, off: number): [number, number, number] =>
    mediapipeToCanonical([data[off], data[off + 1], data[off + 2]]) as [number, number, number];
  const poseWristLm = (data: Float32Array, idx: number): [number, number, number] =>
    mediapipeToCanonical([data[idx * 3], data[idx * 3 + 1], data[idx * 3 + 2]]) as [number, number, number];

  for (const rec of recs) {
    const file = path.join(DATA_DIR, rec.endsWith(".json") ? rec : `${rec}.json`);
    if (!fs.existsSync(file)) { console.log(`!! ${rec} not found`); continue; }
    const j = JSON.parse(fs.readFileSync(file, "utf8")) as { frames: number[] };
    const flat = new Float32Array(j.frames);
    const n = Math.floor(flat.length / FF);

    // first frame with a hand
    let fi = -1;
    for (let i = 0; i < n; i++) {
      const f = flat.slice(i * FF, (i + 1) * FF);
      const pose = mediapipeToCanonicalPose(f, i / 24);
      if (pose && lastHandAssignment && (lastHandAssignment.left !== null || lastHandAssignment.right !== null)) { fi = i; break; }
    }
    if (fi < 0) { console.log(`${rec}: no hand frames`); continue; }

    const frame = flat.slice(fi * FF, (fi + 1) * FF);
    const pose = mediapipeToCanonicalPose(frame, fi / 24)!;
    const assign = lastHandAssignment!;
    console.log(`\n=== ${rec} frame ${fi} ===`);
    console.log(`  assignment: left<->slot ${assign.left}   right<->slot ${assign.right}`);

    reset();
    const result = retarget(sk, pose);
    applyBoneRotations(result.boneRotations, sk);
    scene.updateMatrixWorld(true);

    // rendered shoulder frame
    const lArm = wc(sk.getBone("leftUpperArm" as any)?.bone)!;
    const rArm = wc(sk.getBone("rightUpperArm" as any)?.bone)!;
    const sMid: [number, number, number] = [(lArm[0]+rArm[0])/2, (lArm[1]+rArm[1])/2, (lArm[2]+rArm[2])/2];
    const sDist = Math.max(dist(lArm, rArm), 1e-6);
    const toNorm = (p: [number, number, number]): [number, number, number] => [
      (p[0]-sMid[0])/sDist, (p[1]-sMid[1])/sDist, (p[2]-sMid[2])/sDist,
    ];

    const lmSlot99 = handWristLm(frame, LEFT_SLOT);
    const lmSlot162 = handWristLm(frame, RIGHT_SLOT);
    const lmPoseL = poseWristLm(frame, 15);
    const lmPoseR = poseWristLm(frame, 16);

    for (const side of ["left", "right"] as const) {
      const handBone = wc(sk.getBone((side === "left" ? "leftHand" : "rightHand") as any)?.bone);
      if (!handBone) continue;
      const rw = toNorm(handBone);
      const cands: [string, number][] = [
        ["slot99 wrist", dist(rw, lmSlot99)],
        ["slot162 wrist", dist(rw, lmSlot162)],
        ["poseWrist15", dist(rw, lmPoseL)],
        ["poseWrist16", dist(rw, lmPoseR)],
      ];
      cands.sort((a, b) => a[1] - b[1]);
      console.log(
        `  rendered ${side.padEnd(5)} hand @ (${rw.map(v=>v.toFixed(2)).join(",")})  ` +
        `closest: ${cands[0][0]} d=${cands[0][1].toFixed(3)}  |  ` +
        `slot99 ${cands.find(c=>c[0]==="slot99 wrist")![1].toFixed(2)}  ` +
        `slot162 ${cands.find(c=>c[0]==="slot162 wrist")![1].toFixed(2)}  ` +
        `poseL ${cands.find(c=>c[0]==="poseWrist15")![1].toFixed(2)}  ` +
        `poseR ${cands.find(c=>c[0]==="poseWrist16")![1].toFixed(2)}`
      );
    }
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
