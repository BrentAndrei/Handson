import fs from "fs";
import path from "path";

const glbPath = path.join(import.meta.dirname, "public/models/Xbot.glb");
const buffer = fs.readFileSync(glbPath);
const jsonStr = buffer.toString("utf8", 20, 20 + new DataView(buffer.buffer).getUint32(12, true));
const gltf = JSON.parse(jsonStr);

// Build parent map
const nodeParentMap = new Map<number, number>();
for (const [i, node] of gltf.nodes.entries()) {
  if (node.children) {
    for (const childIdx of node.children) {
      nodeParentMap.set(childIdx, i);
    }
  }
}

const BONE_MAP: Record<string, string> = {
  Hips: "root", Spine: "torso", Head: "head", Neck: "neck",
  LeftShoulder: "leftShoulder", LeftArm: "leftUpperArm",
  LeftForeArm: "leftForearm", LeftHand: "leftHand",
  RightShoulder: "rightShoulder", RightArm: "rightUpperArm",
  RightForeArm: "rightForearm", RightHand: "rightHand",
};

const boneLocalQuats: Record<string, [number, number, number, number]> = {};
const boneLocalPos: Record<string, [number, number, number]> = {};

for (const [i, node] of gltf.nodes.entries()) {
  if (!node.name || !node.name.startsWith("mixamorig:")) continue;
  const simpleName = node.name.replace("mixamorig:", "");
  const mapped = BONE_MAP[simpleName];
  if (mapped) {
    boneLocalQuats[mapped] = node.rotation || [0, 0, 0, 1];
    boneLocalPos[mapped] = node.translation || [0, 0, 0];
  }
}

console.log("=== Bone LOCAL positions (relative to parent) ===");
for (const key of ["leftShoulder", "leftUpperArm", "leftForearm", "leftHand",
                   "rightShoulder", "rightUpperArm", "rightForearm", "rightHand"]) {
  const pos = boneLocalPos[key];
  console.log(key + ": [" + pos.map(v => v.toFixed(4)).join(", ") + "]");
}

console.log("\n=== Bone LOCAL rotations ===");
for (const key of ["leftShoulder", "leftUpperArm", "leftForearm", "leftHand",
                   "rightShoulder", "rightUpperArm", "rightForearm", "rightHand"]) {
  const rot = boneLocalQuats[key];
  console.log(key + ": [" + rot.map(v => v.toFixed(6)).join(", ") + "]");
}

// Now test retargeting with ADOPT.json data
const adoptData = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, "public/avatar-data/ADOPT.json"), "utf8"));
const floatData = new Float32Array(adoptData.frames.flat());

const POSE_COUNT = 33;
const HAND_COUNT = 21;
const poseOffset = 0;
const leftHandOffset = POSE_COUNT * 3;
const rightHandOffset = (POSE_COUNT + HAND_COUNT) * 3;

function getData(buf: Float32Array, offset: number, index: number): [number, number, number] {
  const base = offset + index * 3;
  return [buf[base], buf[base + 1], buf[base + 2]];
}

function vSub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vLen(v: [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function normalizeVec3(v: [number, number, number]): [number, number, number] {
  const len = vLen(v);
  if (len < 1e-8) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function mediapipeToGltf(v: [number, number, number]): [number, number, number] {
  return [v[0], -v[1], -v[2]];
}

function alignBoneToDirection(
  defaultBoneDir: [number, number, number],
  targetDir: [number, number, number],
): [number, number, number, number] {
  const target = normalizeVec3(targetDir);
  if (target[0] === 0 && target[1] === 0 && target[2] === 0) return [0, 0, 0, 1];

  const dx = defaultBoneDir[0];
  const dy = defaultBoneDir[1];
  const dz = defaultBoneDir[2];
  const tx = target[0];
  const ty = target[1];
  const tz = target[2];

  const dot = dx * tx + dy * ty + dz * tz;

  if (dot < -0.99999) {
    return [0, 0, 0, 1];
  }

  const s = Math.sqrt((1 + dot) * 2);
  const invS = s < 1e-8 ? 0 : 1 / s;
  const cx = dy * tz - dz * ty;
  const cy = dz * tx - dx * tz;
  const cz = dx * ty - dy * tx;

  return [cx * invS, cy * invS, cz * invS, s * 0.5];
}

function quatStr(q: [number, number, number, number]): string {
  return "[" + q.map(v => Number(v.toFixed(6))).join(", ") + "]";
}

// Get landmark data for frame 0
const leftShoulder = getData(floatData, poseOffset, 11);
const rightShoulder = getData(floatData, poseOffset, 12);
const leftElbow = getData(floatData, poseOffset, 13);
const rightElbow = getData(floatData, poseOffset, 14);
const leftWrist = getData(floatData, poseOffset, 15);
const rightWrist = getData(floatData, poseOffset, 16);
const leftHandMcp = getData(floatData, leftHandOffset, 9);
const rightHandMcp = getData(floatData, rightHandOffset, 9);

// Directions (MediaPipe space)
const leftElbowDir = vSub(leftElbow, leftShoulder);
const rightElbowDir = vSub(rightElbow, rightShoulder);
const leftForearmDir = vSub(leftWrist, leftElbow);
const rightForearmDir = vSub(rightWrist, rightElbow);
const leftHandDir = vSub(leftHandMcp, leftWrist);
const rightHandDir = vSub(rightHandMcp, rightWrist);

// Convert to GLTF (flip Y, Z - no parent transform since parent quats are identity)
console.log("\n=== FIXED: Delta quaternions with corrected default bone directions ===");
console.log("Left upper arm (default [1,0,0]):", quatStr(alignBoneToDirection([1, 0, 0], mediapipeToGltf(leftElbowDir))));
console.log("Right upper arm (default [-1,0,0]):", quatStr(alignBoneToDirection([-1, 0, 0], mediapipeToGltf(rightElbowDir))));
console.log("Left forearm (default [1,0,0]):", quatStr(alignBoneToDirection([1, 0, 0], mediapipeToGltf(leftForearmDir))));
console.log("Right forearm (default [-1,0,0]):", quatStr(alignBoneToDirection([-1, 0, 0], mediapipeToGltf(rightForearmDir))));
console.log("Left hand (default [1,0,0]):", quatStr(alignBoneToDirection([1, 0, 0], mediapipeToGltf(leftHandDir))));
console.log("Right hand (default [-1,0,0]):", quatStr(alignBoneToDirection([-1, 0, 0], mediapipeToGltf(rightHandDir))));

console.log("\n=== OLD (WRONG): Delta quaternions with swapped default bone directions ===");
console.log("Left upper arm (old [-1,0,0]):", quatStr(alignBoneToDirection([-1, 0, 0], mediapipeToGltf(leftElbowDir))));
console.log("Right upper arm (old [1,0,0]):", quatStr(alignBoneToDirection([1, 0, 0], mediapipeToGltf(rightElbowDir))));

function quatMag(q: [number, number, number, number]): number {
  return Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
}
console.log("\n=== Quaternion magnitude verification ===");
console.log("Left upper arm:", quatMag(alignBoneToDirection([1, 0, 0], mediapipeToGltf(leftElbowDir))).toFixed(6));
console.log("Right upper arm:", quatMag(alignBoneToDirection([-1, 0, 0], mediapipeToGltf(rightElbowDir))).toFixed(6));
console.log("Left forearm:", quatMag(alignBoneToDirection([1, 0, 0], mediapipeToGltf(leftForearmDir))).toFixed(6));
console.log("Right forearm:", quatMag(alignBoneToDirection([-1, 0, 0], mediapipeToGltf(rightForearmDir))).toFixed(6));
console.log("Left hand:", quatMag(alignBoneToDirection([1, 0, 0], mediapipeToGltf(leftHandDir))).toFixed(6));
console.log("Right hand:", quatMag(alignBoneToDirection([-1, 0, 0], mediapipeToGltf(rightHandDir))).toFixed(6));

// Also test with the actual retargetFrames function
console.log("\n=== Full retargetFrames output (frame 0) ===");
const { retargetFrames } = await import("./src/avatar/retarget.ts");
const frames = retargetFrames(floatData);
for (const key of ["leftUpperArm", "rightUpperArm", "leftForearm", "rightForearm", "leftHand", "rightHand", "leftShoulder", "rightShoulder"]) {
  console.log(key + ":", quatStr(frames[0][key] as [number, number, number, number]));
}

process.exit(0);
