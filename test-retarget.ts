import fs from "fs";
import { retargetFrames } from "./src/avatar/retarget";

const data = JSON.parse(fs.readFileSync("public/avatar-data/ADOPT.json", "utf8"));
const floatData = new Float32Array(data.frames.flat());
const frames = retargetFrames(floatData);

console.log("=== Retargeted frame 0 (WITHOUT parentWorldQuats) ===");
for (const key of ["leftShoulder", "leftUpperArm", "leftForearm", "leftHand", "rightShoulder", "rightUpperArm", "rightForearm", "rightHand"]) {
  console.log(key + ":", JSON.stringify(frames[0][key]));
}

const identity: [number, number, number, number] = [0, 0, 0, 1];
const parentQuats = { leftShoulder: identity, rightShoulder: identity, leftUpperArm: identity, rightUpperArm: identity, leftForearm: identity, rightForearm: identity };
const frames2 = retargetFrames(floatData, parentQuats);

console.log("\n=== Retargeted frame 0 (WITH identity parentWorldQuats) ===");
for (const key of ["leftShoulder", "leftUpperArm", "leftForearm", "leftHand", "rightShoulder", "rightUpperArm", "rightForearm", "rightHand"]) {
  console.log(key + ":", JSON.stringify(frames2[0][key]));
}

console.log("\n=== Comparing arm quats WITH vs WITHOUT parentWorldQuats ===");
for (const key of ["leftUpperArm", "leftForearm", "leftHand", "rightUpperArm", "rightForearm", "rightHand"]) {
  const without = frames[0][key] as number[];
  const with_ = frames2[0][key] as number[];
  const diff = without.map((v, i) => Math.abs(v - with_[i]));
  const maxDiff = Math.max(...diff);
  console.log(key + " max diff:", maxDiff.toFixed(8));
}

console.log("\n=== Raw landmark data (frame 0) ===");
const poseOffset = 0 * 345;
const POSE_COUNT = 33;
const HAND_COUNT = 21;
const leftHandOffset = poseOffset + POSE_COUNT * 3;
const rightHandOffset = poseOffset + (POSE_COUNT + HAND_COUNT) * 3;

function getData(buf: Float32Array, offset: number, index: number) {
  const base = offset + index * 3;
  return [buf[base], buf[base+1], buf[base+2]];
}

const ls = getData(floatData, poseOffset, 11);
const rs = getData(floatData, poseOffset, 12);
const le = getData(floatData, poseOffset, 13);
const re = getData(floatData, poseOffset, 14);
const lw = getData(floatData, poseOffset, 15);
const rw = getData(floatData, poseOffset, 16);
const lmcp = getData(floatData, leftHandOffset, 9);
const rmcp = getData(floatData, rightHandOffset, 9);

console.log("leftShoulder (landmark 11):", JSON.stringify(ls));
console.log("rightShoulder (landmark 12):", JSON.stringify(rs));
console.log("leftElbow (landmark 13):", JSON.stringify(le));
console.log("rightElbow (landmark 14):", JSON.stringify(re));
console.log("leftWrist (landmark 15):", JSON.stringify(lw));
console.log("rightWrist (landmark 16):", JSON.stringify(rw));
console.log("leftHandMCP (landmark 9):", JSON.stringify(lmcp));
console.log("rightHandMCP (landmark 9):", JSON.stringify(rmcp));

// Direction analysis
const leDir = [le[0]-ls[0], le[1]-ls[1], le[2]-ls[2]];
const reDir = [re[0]-rs[0], re[1]-rs[1], re[2]-rs[2]];
console.log("\nleftElbowDir (shoulder->elbow, MediaPipe):", JSON.stringify(leDir));
console.log("rightElbowDir (shoulder->elbow, MediaPipe):", JSON.stringify(reDir));

// GLTF conversion (flip Y, Z)
const leGltd = [leDir[0], -leDir[1], -leDir[2]];
const reGltd = [reDir[0], -reDir[1], -reDir[2]];
console.log("leftElbowDir (GLTF):", JSON.stringify(leGltd));
console.log("rightElbowDir (GLTF):", JSON.stringify(reGltd));

// Normalize
const leLen = Math.hypot(...leGltd);
const leNorm = leGltd.map(v => v/leLen);
const reLen = Math.hypot(...reGltd);
const reNorm = reGltd.map(v => v/reLen);
console.log("leftElbowDir normalized (GLTF):", JSON.stringify(leNorm.map(v => Number(v.toFixed(6)))));
console.log("rightElbowDir normalized (GLTF):", JSON.stringify(reNorm.map(v => Number(v.toFixed(6)))));

// Hand direction analysis
const lhDir = [lmcp[0]-lw[0], lmcp[1]-lw[1], lmcp[2]-lw[2]];
const rhDir = [rmcp[0]-rw[0], rmcp[1]-rw[1], rmcp[2]-rw[2]];
console.log("\nleftHandDir (wrist->MCP, MediaPipe):", JSON.stringify(lhDir));
console.log("rightHandDir (wrist->MCP, MediaPipe):", JSON.stringify(rhDir));
const lhGltd = [lhDir[0], -lhDir[1], -lhDir[2]];
const rhGltd = [rhDir[0], -rhDir[1], -rhDir[2]];
console.log("leftHandDir (GLTF):", JSON.stringify(lhGltd));
console.log("rightHandDir (GLTF):", JSON.stringify(rhGltd));
