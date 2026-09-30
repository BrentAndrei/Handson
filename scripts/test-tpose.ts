/**
 * Synthetic T-pose retargeting test (Phase 2).
 *
 * Creates a canonical T-pose matching the Xbot avatar's bind pose and
 * verifies that running it through the retargeter produces near-identity
 * rotations.
 *
 * Run with: npx tsx scripts/test-tpose.ts
 */
import {
  Bone,
  Matrix4,
  Quaternion,
  Vector3,
} from "three";
import { retarget } from "../src/avatar/retargeter";
import { ResolvedSkeleton, ResolvedBone, CanonicalRole } from "../src/avatar/boneResolver";
import { CanonicalPose, BodyPose, HandPose, Vec3, toVec3 } from "../src/avatar/canonicalPose";
import { canonicalToGltf, gltfToCanonical, Coords } from "../src/avatar/coordinateSystem";

// ---------------------------------------------------------------------------
// Bone data from skeleton-report.txt (GLTF localPos in cm, identity rotation)
// ---------------------------------------------------------------------------

interface BoneData {
  name: string;
  parent: string | null;
  localPos: [number, number, number];
  localRot: [number, number, number, number];
}

const BONE_DATA: BoneData[] = [
  { name: "Armature", parent: null, localPos: [0, 0, 0], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:Hips", parent: "Armature", localPos: [0, 103.99, 2.08], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:Spine", parent: "mixamorig:Hips", localPos: [0, 10.18, 0.13], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:Spine1", parent: "mixamorig:Spine", localPos: [0, 10.08, -1.00], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:Spine2", parent: "mixamorig:Spine1", localPos: [0, 9.10, -1.37], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:Neck", parent: "mixamorig:Spine2", localPos: [0, 16.67, -2.52], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:Head", parent: "mixamorig:Neck", localPos: [0, 9.62, 1.69], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:LeftShoulder", parent: "mixamorig:Spine2", localPos: [4.57, 10.95, -2.63], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:LeftArm", parent: "mixamorig:LeftShoulder", localPos: [10.59, -0.52, -2.23], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:LeftForeArm", parent: "mixamorig:LeftArm", localPos: [27.84, 0, 0], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:LeftHand", parent: "mixamorig:LeftForeArm", localPos: [28.33, 0, 0], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:RightShoulder", parent: "mixamorig:Spine2", localPos: [-4.57, 10.95, -2.63], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:RightArm", parent: "mixamorig:RightShoulder", localPos: [-10.59, -0.52, -2.23], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:RightForeArm", parent: "mixamorig:RightArm", localPos: [-27.84, 0, 0], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:RightHand", parent: "mixamorig:RightForeArm", localPos: [-28.33, 0, 0], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:LeftUpLeg", parent: "mixamorig:Hips", localPos: [8.21, -6.75, -1.60], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:LeftLeg", parent: "mixamorig:LeftUpLeg", localPos: [0, -44.37, 0.28], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:LeftFoot", parent: "mixamorig:LeftLeg", localPos: [0, -44.43, -2.98], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:RightUpLeg", parent: "mixamorig:Hips", localPos: [-8.21, -6.75, -1.60], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:RightLeg", parent: "mixamorig:RightUpLeg", localPos: [0, -44.37, 0.29], localRot: [0, 0, 0, 1] },
  { name: "mixamorig:RightFoot", parent: "mixamorig:RightLeg", localPos: [0, -44.43, -2.98], localRot: [0, 0, 0, 1] },
];

const ARMATURE_SCALE = 0.01;

// ---------------------------------------------------------------------------
// Build bone hierarchy and skeleton
// ---------------------------------------------------------------------------

function buildSkeleton(): { skeleton: ResolvedSkeleton; bones: Map<string, Bone> } {
  const boneObjs = new Map<string, Bone>();
  const boneRecords = new Map<string, ResolvedBone>();

  // Create bone objects
  for (const bd of BONE_DATA) {
    const bone = new Bone();
    bone.name = bd.name;
    bone.position.set(
      bd.localPos[0] * ARMATURE_SCALE,
      bd.localPos[1] * ARMATURE_SCALE,
      bd.localPos[2] * ARMATURE_SCALE,
    );
    bone.quaternion.identity();
    boneObjs.set(bd.name, bone);
  }

  // Set up parent-child hierarchy
  const root = boneObjs.get("Armature")!;
  for (const bd of BONE_DATA) {
    if (bd.name === "Armature") continue;
    const bone = boneObjs.get(bd.name)!;
    const parent = boneObjs.get(bd.parent!)!;
    parent.add(bone);
  }
  root.updateMatrixWorld(true);

  // Build bone records
  for (const bd of BONE_DATA) {
    const bone = boneObjs.get(bd.name)!;
    const children = BONE_DATA.filter((b) => b.parent === bd.name).map((b) => b.name);

    // Compute localDir from first child's localPos
    let localDir = new Vector3(0, 0, 0);
    let length = 0;
    if (children.length > 0) {
      const child = BONE_DATA.find((b) => b.name === children[0])!;
      localDir.set(
        child.localPos[0] * ARMATURE_SCALE,
        child.localPos[1] * ARMATURE_SCALE,
        child.localPos[2] * ARMATURE_SCALE,
      );
      length = localDir.length();
      if (length > 1e-6) localDir.normalize();
    }

    // World position and rotation from matrixWorld
    const worldPos = new Vector3();
    const worldRot = new Quaternion();
    const scale = new Vector3();
    bone.matrixWorld.decompose(worldPos, worldRot, scale);

    const localPos = new Vector3(
      bd.localPos[0] * ARMATURE_SCALE,
      bd.localPos[1] * ARMATURE_SCALE,
      bd.localPos[2] * ARMATURE_SCALE,
    );

    const localRot = new Quaternion(bd.localRot[0], bd.localRot[1], bd.localRot[2], bd.localRot[3]);

    boneRecords.set(bd.name, {
      name: bd.name,
      bone: bone,
      index: BONE_DATA.findIndex((b) => b.name === bd.name),
      parent: bd.parent,
      children,
      localDir,
      length,
      localPos,
      localRot,
      worldPos,
      worldRot,
      avatarForward: localDir.clone(),
    } as ResolvedBone);
  }

  // Build bone mapping (same logic as boneResolver.ts)
  const boneMapping: Record<string, string> = {};
  const findByName = (pattern: string | RegExp): string | undefined => {
    for (const name of boneRecords.keys()) {
      const lower = name.toLowerCase();
      const stripped = lower.replace(/mixamorig[:\-_]?/g, "").replace(/[_-]/g, "");
      if (pattern instanceof RegExp) {
        if (pattern.test(lower) || pattern.test(stripped)) return name;
      } else {
        if (stripped.includes(pattern.toLowerCase().replace(/[_-]/g, ""))) return name;
      }
    }
    return undefined;
  };

  let n: string | undefined;
  n = findByName(/^hips$/); if (n) boneMapping.hips = n;
  const spineNames = Array.from(boneRecords.keys())
    .filter((n) => /spine/i.test(n.toLowerCase().replace("mixamorig", "")))
    .sort((a, b) => {
      const ai = a.toLowerCase().replace("mixamorig", "").match(/spine(\d+)/);
      const bi = b.toLowerCase().replace("mixamorig", "").match(/spine(\d+)/);
      return (ai ? parseInt(ai[1]) : 0) - (bi ? parseInt(bi[1]) : 0);
    });
  if (spineNames.length > 0) boneMapping.spine1 = spineNames[0];
  if (spineNames.length > 1) boneMapping.spine2 = spineNames[1];
  if (spineNames.length > 2) boneMapping.spine3 = spineNames[2];
  n = findByName(/neck/i); if (n) boneMapping.neck = n;
  n = findByName(/^head$/i); if (n) boneMapping.head = n;
  n = findByName(/leftshoulder|leftclavicle/i); if (n) boneMapping.leftShoulder = n;
  n = findByName(/rightshoulder|rightclavicle/i); if (n) boneMapping.rightShoulder = n;
  n = findByName(/leftarm$/); if (n) boneMapping.leftUpperArm = n;
  n = findByName(/rightarm$/); if (n) boneMapping.rightUpperArm = n;
  n = findByName(/leftforearm/i); if (n) boneMapping.leftForearm = n;
  n = findByName(/rightforearm/i); if (n) boneMapping.rightForearm = n;
  n = findByName(/lefthand/i); if (n) boneMapping.leftHand = n;
  n = findByName(/righthand/i); if (n) boneMapping.rightHand = n;
  n = findByName(/leftupleg|lefthip/i); if (n) boneMapping.leftHip = n;
  n = findByName(/rightupleg|righthip/i); if (n) boneMapping.rightHip = n;
  n = findByName(/leftleg|leftknee/i); if (n) boneMapping.leftKnee = n;
  n = findByName(/rightleg|rightknee/i); if (n) boneMapping.rightKnee = n;
  n = findByName(/leftfoot|leftankle/i); if (n) boneMapping.leftAnkle = n;
  n = findByName(/rightfoot|rightankle/i); if (n) boneMapping.rightAnkle = n;

  const skeleton: ResolvedSkeleton = {
    bones: boneRecords as any,
    rootBone: "Armature",
    boneOrder: BONE_DATA.map((b) => b.name),
    armatureScale: ARMATURE_SCALE,
    boneMapping,
    getBone: (role: CanonicalRole) => {
      const boneName = boneMapping[role as string];
      return boneName ? boneRecords.get(boneName) : undefined;
    },
    getWorldPos: (role: CanonicalRole) => {
      const boneName = boneMapping[role as string];
      const bone = boneName ? boneRecords.get(boneName) : undefined;
      return bone ? bone.worldPos : undefined;
    },
    getLocalDir: (role: CanonicalRole) => {
      const boneName = boneMapping[role as string];
      const bone = boneName ? boneRecords.get(boneName) : undefined;
      return bone ? bone.localDir.clone() : undefined;
    },
  } as any;

  return { skeleton, bones: boneObjs };
}

// ---------------------------------------------------------------------------
// Create synthetic T-pose in canonical space (matching avatar bind pose)
// ---------------------------------------------------------------------------

function gltfToCanon(v: Vec3): Vec3 {
  const g = gltfToCanonical(v as Coords);
  return [g[0], g[1], g[2]] as Vec3;
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function normalize(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (l < 1e-10) return [0, 0, 0];
  return [a[0] / l, a[1] / l, a[2] / l];
}

function makeTpose(skeleton: ResolvedSkeleton): CanonicalPose {
   // Compute canonical world positions for each joint
   const canonPos = new Map<string, Vec3>();
   for (const [name, bone] of skeleton.bones) {
     const gltfPos = [bone.worldPos.x, bone.worldPos.y, bone.worldPos.z];
     canonPos.set(name, gltfToCanon(gltfPos as Vec3));
   }

   // Helper: convert a bone's GLTF localDir → canonical Vec3
   function boneCanonDir(bone: ResolvedBone | undefined): Vec3 {
     if (!bone || bone.localDir.length() < 1e-6) return [0, 0, 0];
     return gltfToCanon([bone.localDir.x, bone.localDir.y, bone.localDir.z]);
   }

    // spineDir: canonical up — matches the spine bones' near-vertical direction
    const spineDir: Vec3 = [0, 1, 0];

    // shoulderAxis: R→L in canonical space (canonical -Z)
    const shoulderAxis: Vec3 = [0, 0, -1];

    // headDir: from neck bone's bind direction (neck→head) in canonical space
    const headDir = boneCanonDir(skeleton.getBone("neck" as CanonicalRole));

   const body: BodyPose = {
     hipsPos: canonPos.get(skeleton.boneMapping["hips"])!,
     shoulderHeight: canonPos.get(skeleton.boneMapping["leftShoulder"])![1],
     spineDir,
     shoulderAxis,
     leftShoulder: canonPos.get(skeleton.boneMapping["leftShoulder"])!,
     leftElbow: canonPos.get(skeleton.boneMapping["leftUpperArm"])!,
     leftWrist: canonPos.get(skeleton.boneMapping["leftForearm"])!,
     rightShoulder: canonPos.get(skeleton.boneMapping["rightShoulder"])!,
     rightElbow: canonPos.get(skeleton.boneMapping["rightUpperArm"])!,
     rightWrist: canonPos.get(skeleton.boneMapping["rightForearm"])!,
     neckPos: canonPos.get(skeleton.boneMapping["neck"])!,
     headPos: canonPos.get(skeleton.boneMapping["head"])!,
     headDir,
     leftHip: canonPos.get(skeleton.boneMapping["leftHip"])!,
     leftKnee: canonPos.get(skeleton.boneMapping["leftKnee"])!,
     leftAnkle: canonPos.get(skeleton.boneMapping["leftAnkle"])!,
     rightHip: canonPos.get(skeleton.boneMapping["rightHip"])!,
     rightKnee: canonPos.get(skeleton.boneMapping["rightKnee"])!,
     rightAnkle: canonPos.get(skeleton.boneMapping["rightAnkle"])!,
   };

   // Hand poses: palmDir = hand bone's canonical bind direction.
   // In canonical space, left hand points -Z (character's left), right hand points +Z (character's right)
   // Palm normal = -Y (palms face down in T-pose)
   const leftHandBone = skeleton.getBone("leftHand" as CanonicalRole);
   const rightHandBone = skeleton.getBone("rightHand" as CanonicalRole);

   const leftPalmDir = boneCanonDir(leftHandBone) || [0, 0, -1] as Vec3;
   const rightPalmDir = boneCanonDir(rightHandBone) || [0, 0, 1] as Vec3;
   const palmNormal: Vec3 = [0, -1, 0];

   const leftHandPose: HandPose = {
     wristPos: canonPos.get(skeleton.boneMapping["leftHand"])!,
     palmNormal,
     palmDir: leftPalmDir,
     fingers: {
       thumb: { dir: leftPalmDir, curl: 0 },
       index: { dir: leftPalmDir, curl: 0 },
       middle: { dir: leftPalmDir, curl: 0 },
       ring: { dir: leftPalmDir, curl: 0 },
       pinky: { dir: leftPalmDir, curl: 0 },
     },
   };

   const rightHandPose: HandPose = {
     wristPos: canonPos.get(skeleton.boneMapping["rightHand"])!,
     palmNormal,
     palmDir: rightPalmDir,
     fingers: {
       thumb: { dir: rightPalmDir, curl: 0 },
       index: { dir: rightPalmDir, curl: 0 },
       middle: { dir: rightPalmDir, curl: 0 },
       ring: { dir: rightPalmDir, curl: 0 },
       pinky: { dir: rightPalmDir, curl: 0 },
     },
   };

   return {
     body,
     leftHand: leftHandPose,
     rightHand: rightHandPose,
     timestamp: 0,
   };
 }

// ---------------------------------------------------------------------------
// Direction verification
// ---------------------------------------------------------------------------

function computeWorldBoneDir(bone: ResolvedBone): Vector3 {
  if (!bone.children || bone.children.length === 0) return new Vector3(0, 0, 0);
  const child = bone.bone.children[0] as Bone;
  if (!child) return new Vector3(0, 0, 0);
  const childWorldPos = new Vector3();
  child.getWorldPosition(childWorldPos);
  const parentWorldPos = new Vector3();
  bone.bone.getWorldPosition(parentWorldPos);
  return new Vector3().subVectors(childWorldPos, parentWorldPos).normalize();
}

function dot(a: Vector3, b: Vector3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function quatDot(a: Quaternion, b: Quaternion): number {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

function main() {
  console.log("=== Phase 2: Synthetic T-Pose Test ===\n");

  const { skeleton, bones } = buildSkeleton();
  console.log("Skeleton: " + skeleton.bones.size + " bones, root = " + skeleton.rootBone);
  console.log("Bone mapping keys:", Object.keys(skeleton.boneMapping).join(", "));

  // --- Print bind directions ---
  console.log("\n--- Bind Directions (GLTF) ---");
  const bodyChain: [string, string, string][] = [
    ["spine1", "spine2", "spine3"],
    ["neck", "head", ""],
    ["leftShoulder", "leftUpperArm", "leftForearm"],
    ["rightShoulder", "rightUpperArm", "rightForearm"],
    ["leftHip", "leftKnee", "leftAnkle"],
    ["rightHip", "rightKnee", "rightAnkle"],
  ];

  for (const [r1, r2, r3] of bodyChain) {
    for (const role of [r1, r2, r3]) {
      if (!role) continue;
      const bone = skeleton.getBone(role as CanonicalRole);
      if (!bone) {
        console.log(`  ${role}: NOT FOUND`);
        continue;
      }
      console.log(`  ${role} (${bone.name}): GLTF dir=[${bone.localDir.x.toFixed(4)}, ${bone.localDir.y.toFixed(4)}, ${bone.localDir.z.toFixed(4)}] len=${bone.length.toFixed(4)}`);
    }
  }

  // --- Create T-pose ---
  const tpose = makeTpose(skeleton);
  console.log("\n--- Synthetic T-Pose (Canonical) ---");
  const b = tpose.body;
  console.log("hipsPos:", b.hipsPos.map((v) => v.toFixed(4)));
  console.log("spineDir:", b.spineDir.map((v) => v.toFixed(4)));
  console.log("shoulderAxis:", b.shoulderAxis.map((v) => v.toFixed(4)));

  // Check target directions (canonical → GLTF via cvec)
  console.log("\n--- Target Directions (GLTF via cvec) ---");
  const targets: [string, Vec3 | null, Vec3][] = [
    ["leftShoulder→leftElbow", b.leftShoulder, b.leftElbow],
    ["leftElbow→leftWrist", b.leftElbow, b.leftWrist],
    ["rightShoulder→rightElbow", b.rightShoulder, b.rightElbow],
    ["rightElbow→rightWrist", b.rightElbow, b.rightWrist],
    ["leftHip→leftKnee", b.leftHip, b.leftKnee],
    ["leftKnee→leftAnkle", b.leftKnee, b.leftAnkle],
    ["rightHip→rightKnee", b.rightHip, b.rightKnee],
    ["rightKnee→rightAnkle", b.rightKnee, b.rightAnkle],
  ];

  function cvec(v: Vec3): Vector3 {
    const g = canonicalToGltf(v as Coords);
    return new Vector3(g[0], g[1], g[2]);
  }

  for (const [label, from, to] of targets) {
    const fromGltf = cvec(from!);
    const toGltf = cvec(to!);
    const dir = new Vector3().subVectors(toGltf, fromGltf).normalize();
    console.log(`  ${label}: target GLTF=[${dir.x.toFixed(4)}, ${dir.y.toFixed(4)}, ${dir.z.toFixed(4)}]`);
  }

  // --- Run retargeter ---
  console.log("\n--- Running retargeter ---");
  const result = retarget(skeleton as any, tpose);
  console.log("Produced " + result.boneRotations.size + " bone rotations");

  // --- Check quaternion identity ---
  console.log("\n--- Quaternion Identity Check ---");
  const identityQuat = new Quaternion(0, 0, 0, 1);
  let allIdentity = true;
  for (const [name, rot] of result.boneRotations) {
    const d = Math.abs(quatDot(identityQuat, rot));
    const isIdentity = d > 0.999;
    if (!isIdentity) allIdentity = false;
    const angle = Math.acos(Math.min(1, Math.max(-1, d))) * 2 * 180 / Math.PI;
    const status = isIdentity ? "OK" : "FAIL";
    console.log(`  ${name}: q=[${rot.x.toFixed(4)}, ${rot.y.toFixed(4)}, ${rot.z.toFixed(4)}, ${rot.w.toFixed(4)}] dot=${d.toFixed(4)} angle=${angle.toFixed(1)}deg [${status}]`);
  }

  // --- Apply rotations and check bone directions ---
  console.log("\n--- Post-retarget Bone Direction Check ---");
  for (const [name, bone] of skeleton.bones) {
    const rot = result.boneRotations.get(name);
    if (!rot) continue;
    bone.bone.quaternion.copy(rot);
  }
  // Update world matrices
  bones.get("Armature")!.updateMatrixWorld(true);

  let dirMatches = 0;
  let dirTotal = 0;
  const dirChecks: [string, CanonicalRole, Vec3][] = [
    ["leftUpperArm", "leftUpperArm", [1, 0, 0]],  // Should match GLTF bind dir
    ["leftForearm", "leftForearm", [1, 0, 0]],
    ["rightUpperArm", "rightUpperArm", [-1, 0, 0]],
    ["rightForearm", "rightForearm", [-1, 0, 0]],
    ["spine1", "spine1", [0, 1, 0]],
    ["leftHip", "leftHip", [0, -1, 0]], // LeftUpLeg→LeftLeg direction
    ["rightHip", "rightHip", [0, -1, 0]],
    ["leftKnee", "leftKnee", [0, -1, 0]],
    ["rightKnee", "rightKnee", [0, -1, 0]],
  ];

  for (const [label, role, expectedGltf] of dirChecks) {
    dirTotal++;
    const bone = skeleton.getBone(role);
    if (!bone) {
      console.log(`  ${label}: bone not found`);
      continue;
    }
    // Get the actual child bone world direction
    const actualDir = computeWorldBoneDir(bone);
    const expectedDir = new Vector3(expectedGltf[0], expectedGltf[1], expectedGltf[2]).normalize();
    const d = dot(actualDir, expectedDir);
    const ok = d > 0.95;
    if (ok) dirMatches++;
    console.log(`  ${label}: expected=[${expectedDir.x.toFixed(4)}, ${expectedDir.y.toFixed(4)}, ${expectedDir.z.toFixed(4)}] actual=[${actualDir.x.toFixed(4)}, ${actualDir.y.toFixed(4)}, ${actualDir.z.toFixed(4)}] dot=${d.toFixed(4)} [${ok ? "OK" : "FAIL"}]`);
  }

  // --- Summary ---
  console.log("\n=== Summary ===");
  console.log("Quaternion identity: " + (allIdentity ? "PASS" : "FAIL (" + (100*(1-allIdentity)) + "% deviated)"));
  console.log("Direction check: " + (dirMatches === dirTotal ? "PASS" : "FAIL (" + (dirTotal - dirMatches) + "/" + dirTotal + " failed)"));

  const overall = allIdentity && dirMatches === dirTotal;
  console.log("Overall T-pose: " + (overall ? "PASS" : "FAIL"));
}

main();
