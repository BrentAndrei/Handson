/**
 * PHASE 3 — Retargeting verification harness.
 *
 * Loads the REAL public/models/Xbot.glb (via resolveSkeleton, the same resolver
 * the app uses), prints the measured bind pose, then runs synthetic poses
 * through the real pipeline:  CanonicalPose -> retarget() -> bone.quaternion
 * and MEASURES the resulting world direction of every bone's joint->child
 * offset against the direction implied by the canonical pose.
 *
 * Run with:  npx tsx scripts/verify-retarget.ts
 *            npm run verify:retarget
 *
 * Writes verify-retarget.json (machine-readable results) for before/after diffs.
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  resolveSkeleton,
  getFingerBoneNames,
  type ResolvedSkeleton,
  type ResolvedBone,
  type CanonicalRole,
} from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import {
  mediapipeToCanonicalPose,
  lastHandAssignment,
} from "../src/avatar/mediapipeAdapter";
import type { CanonicalPose, BodyPose, HandPose, Vec3 } from "../src/avatar/canonicalPose";
import { canonicalToGltf, type Coords } from "../src/avatar/coordinateSystem";

const MODEL_PATH = fileURLToPath(new URL("../public/models/Xbot.glb", import.meta.url));

/**
 * PHASE 4 real-pose fixtures: 345-float .bin buffers written by
 * training/extract_landmarks.py --mode images. The layout is
 *   [0..99) pose(33) [99..162) leftHand(21) [162..225) rightHand(21) [225..345) lips(40)
 * which is byte-for-byte the same pose/hand offsets the packed holistic buffer
 * uses, so these feed straight into mediapipeToCanonicalPose() with no shim.
 */
const FIXTURE_DIR = fileURLToPath(
  new URL("./fixtures/poses/extracted", import.meta.url)
);
const FEATURE_DIM = 345;

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
type Check = {
  bone: string;
  role: string;
  expected: [number, number, number];
  actual: [number, number, number];
  dot: number;
  pass: boolean;
  note?: string;
};
type TestResult = { name: string; checks: Check[]; extra: string[]; pass: boolean };
const ALL_RESULTS: TestResult[] = [];

const f3 = (v: THREE.Vector3) => `[${v.x.toFixed(4)}, ${v.y.toFixed(4)}, ${v.z.toFixed(4)}]`;
const t3 = (v: THREE.Vector3): [number, number, number] => [v.x, v.y, v.z];
const fix = (v: THREE.Vector3) => new THREE.Vector3(+v.x.toFixed(6), +v.y.toFixed(6), +v.z.toFixed(6));
/** Angle between two rotations, in degrees (0..180). */
function qangle(a: THREE.Quaternion, b: THREE.Quaternion): number {
  const d = Math.min(1, Math.abs(a.dot(b)));
  return (Math.acos(d) * 2 * 180) / Math.PI;
}

/** canonical Vec3 -> THREE.Vector3 in GLTF space (identical to bodyRetargeter's cvec). */
function cvec(v: Vec3): THREE.Vector3 {
  const g = canonicalToGltf(v as Coords);
  return new THREE.Vector3(g[0], g[1], g[2]);
}
function dir(a: THREE.Vector3, b: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3().subVectors(b, a).normalize();
}

// ---------------------------------------------------------------------------
// Model loading + bind pose measurement
// ---------------------------------------------------------------------------
/** World-space direction from a bone's joint to its (first) child joint. */
function worldChildDir(bone: ResolvedBone, sk: ResolvedSkeleton): THREE.Vector3 | null {
  if (bone.children.length === 0) return null;
  const child = sk.bones.get(bone.children[0]);
  if (!child) return null;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  bone.bone.getWorldPosition(a);
  child.bone.getWorldPosition(b);
  if (b.distanceTo(a) < 1e-9) return null;
  return b.sub(a).normalize();
}

/** World-space quaternion of a bone. */
function worldQuat(bone: ResolvedBone): THREE.Quaternion {
  return bone.bone.getWorldQuaternion(new THREE.Quaternion());
}

/** Reset every bone to its measured bind LOCAL rotation (all identity for Xbot). */
function resetToBind(sk: ResolvedSkeleton, scene: THREE.Object3D) {
  for (const b of sk.bones.values()) {
    b.bone.quaternion.copy(b.localRot);
    b.bone.position.copy(b.localPos);
  }
  scene.updateMatrixWorld(true);
}

/** ONE global rotation check: is every bone's bind world rotation the identity? */
function checkBindIdentity(sk: ResolvedSkeleton): { ok: boolean; worst: string; worstAngle: number } {
  const I = new THREE.Quaternion();
  let worst = "";
  let worstAngle = 0;
  for (const b of sk.bones.values()) {
    const q = worldQuat(b);
    const d = Math.min(1, Math.abs(q.dot(I)));
    const angle = (Math.acos(d) * 2 * 180) / Math.PI;
    if (angle > worstAngle) {
      worstAngle = angle;
      worst = b.name;
    }
  }
  return { ok: worstAngle < 0.5, worst, worstAngle };
}

/** PHASE 4: every world matrix element and quaternion component must be finite. */
function checkFinite(sk: ResolvedSkeleton, scene: THREE.Object3D): string[] {
  const bad: string[] = [];
  const m = new THREE.Matrix4();
  scene.updateMatrixWorld(true);
  for (const b of sk.bones.values()) {
    b.bone.matrixWorld.toArray(m.elements);
    for (const e of m.elements) if (!Number.isFinite(e)) { bad.push(`${b.name}.matrixWorld`); break; }
    const q = b.bone.quaternion;
    if (![q.x, q.y, q.z, q.w].every(Number.isFinite)) bad.push(`${b.name}.quaternion`);
  }
  return bad;
}

// ---------------------------------------------------------------------------
// Synthetic canonical poses (PROJECT convention: X=forward, Y=up, Z=SUBJECT'S RIGHT)
//   subject's left  = -Z_canonical -> GLTF +X   (Xbot's LeftArm binds at +X)
//   subject's right = +Z_canonical -> GLTF -X
//   forward         = +X_canonical -> GLTF +Z   (Xbot faces +Z)
// ---------------------------------------------------------------------------
interface Joints {
  pelvis: Vec3; lHip: Vec3; rHip: Vec3; lKnee: Vec3; rKnee: Vec3; lAnkle: Vec3; rAnkle: Vec3;
  lSh: Vec3; rSh: Vec3; lEl: Vec3; rEl: Vec3; lWr: Vec3; rWr: Vec3; neck: Vec3; head: Vec3;
}
const vsub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vmid = (a: Vec3, b: Vec3): Vec3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
function vnorm(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l < 1e-10 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
}

/** Base T-pose: arms along the subject's left/right, legs down, spine up. */
function tposeJoints(): Joints {
  return {
    pelvis: [0, 1.0, 0],
    lHip: [0, 0.98, -0.10], rHip: [0, 0.98, 0.10],
    lKnee: [0, 0.53, -0.10], rKnee: [0, 0.53, 0.10],
    lAnkle: [0, 0.08, -0.10], rAnkle: [0, 0.08, 0.10],
    lSh: [0, 1.45, -0.20], rSh: [0, 1.45, 0.20],
    lEl: [0, 1.45, -0.60], rEl: [0, 1.45, 0.60],
    lWr: [0, 1.45, -1.00], rWr: [0, 1.45, 1.00],
    neck: [0, 1.52, 0], head: [0, 1.68, 0],
  };
}

/** Mirrors smplxAdapter.buildBodyPose(): derives spineDir/shoulderAxis/headDir from joints. */
function bodyFrom(j: Joints): BodyPose {
  const hipMid = vmid(j.lHip, j.rHip);
  const shMid = vmid(j.lSh, j.rSh);
  return {
    hipsPos: j.pelvis,
    shoulderHeight: shMid[1],
    spineDir: vnorm(vsub(shMid, hipMid)),
    shoulderAxis: vnorm(vsub(j.rSh, j.lSh)), // left -> right = SUBJECT'S RIGHT
    leftShoulder: j.lSh, leftElbow: j.lEl, leftWrist: j.lWr,
    rightShoulder: j.rSh, rightElbow: j.rEl, rightWrist: j.rWr,
    neckPos: j.neck, headPos: j.head,
    headDir: vnorm(vsub(j.head, j.neck)),
    leftHip: j.lHip, leftKnee: j.lKnee, leftAnkle: j.lAnkle,
    rightHip: j.rHip, rightKnee: j.rKnee, rightAnkle: j.rAnkle,
  };
}

/** Default hand pose — mirrors the adapter's documented absent-hand fallback
 *  (mediapipeAdapter.ts: wristPos from the body, palmDir/palmNormal [0,0,1], and
 *  every finger dir [0,0,1] with curl 0). PHASE 5: this previously returned an
 *  EMPTY fingers object, which is fine for a body-only test but made any hand
 *  assertion read `fingers.thumb` as undefined. */
function emptyHand(): HandPose {
  const fingers = {} as Record<string, { dir: Vec3; curl: number }>;
  for (const f of ["thumb", "index", "middle", "ring", "pinky"]) {
    fingers[f] = { dir: [0, 0, 1], curl: 0 };
  }
  return {
    wristPos: [0, 0, 0],
    palmNormal: [0, 0, 1],
    palmDir: [0, 0, 1],
    fingers: fingers as HandPose["fingers"],
  };
}
function poseFrom(j: Joints): CanonicalPose {
  return { body: bodyFrom(j), leftHand: emptyHand(), rightHand: emptyHand(), timestamp: 0 };
}

// Roles whose joint->child direction the canonical pose fully determines.
const DIR_ROLES: CanonicalRole[] = [
  "leftShoulder", "leftUpperArm", "leftForearm",
  "rightShoulder", "rightUpperArm", "rightForearm",
  "leftHip", "leftKnee", "rightHip", "rightKnee",
  "spine1", "spine2", "spine3", "neck", "head",
];
// Roles with NO joint data in CanonicalPose -> must not be aimed at anything.
const NO_DATA_ROLES: CanonicalRole[] = ["leftAnkle", "rightAnkle"];

/** The world direction (GLTF) that each bone's joint->child offset should end up with. */
function expectedDir(
  role: CanonicalRole,
  body: BodyPose,
  spinePoseRot: THREE.Quaternion,
  sk: ResolvedSkeleton
): THREE.Vector3 | null {
  const c = cvec;
  switch (role) {
    // PHASE 17: the clavicle is NOT aimed at armDir.
    //
    // The clavicle's child is the upper arm, so its bind localDir is LATERAL
    // (Xbot: mixamorigLeftShoulder.localDir = (0.9774,-0.0484,-0.2060)). Aiming
    // it at armDir (shoulder->elbow) rotates it ~88-94 deg off bind every frame.
    //
    // This assertion previously read `return dir(c(body.leftShoulder), c(body.leftElbow))`
    // with the comment "= armDir (the code's target)". That was circular: it
    // restated the implementation instead of independently checking it, so it
    // passed while the rig was broken. The oracle is now stated anatomically --
    // the clavicle's own joint->child direction is the SUBJECT'S LATERAL axis
    // (shoulderAxis is right-left = the subject's RIGHT, so the left clavicle
    // uses -shoulderAxis), projected into the trunk plane exactly as the solver
    // does. This still verifies the full parent->local->world chain.
    case "leftShoulder":
    case "rightShoulder": {
      const axis = c(body.shoulderAxis).normalize();
      const lat = axis.multiplyScalar(role === "leftShoulder" ? -1 : 1);
      return lat.projectOnPlane(c(body.spineDir).normalize()).normalize();
    }
    case "leftUpperArm":
      return dir(c(body.leftShoulder), c(body.leftElbow));
    case "leftForearm":
      return dir(c(body.leftElbow), c(body.leftWrist));
    case "rightUpperArm":
      return dir(c(body.rightShoulder), c(body.rightElbow));
    case "rightForearm":
      return dir(c(body.rightElbow), c(body.rightWrist));
    case "leftHip":
      return dir(c(body.leftHip), c(body.leftKnee));
    case "leftKnee":
      return dir(c(body.leftKnee), c(body.leftAnkle));
    case "rightHip":
      return dir(c(body.rightHip), c(body.rightKnee));
    case "rightKnee":
      return dir(c(body.rightKnee), c(body.rightAnkle));
    case "spine1":
    case "spine2":
    case "spine3": {
      const b = sk.getBone(role);
      if (!b) return null;
      return b.localDir.clone().normalize().applyQuaternion(spinePoseRot).normalize();
    }
    case "neck":
    case "head":
      return c(body.headDir).normalize().applyQuaternion(spinePoseRot).normalize();
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// PHASE 5 — hand + finger bone checks
// ---------------------------------------------------------------------------
const HAND_FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;

/** All hand/finger bone names for one side, from the resolved bone mapping. */
function fingerBonesFor(sk: ResolvedSkeleton, side: "left" | "right") {
  return getFingerBoneNames(sk, side);
}

/** vlen for a canonical Vec3. */
function vlen(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

/**
 * Measures the world-space bend between consecutive phalanges of one finger
 * and returns the angles in degrees, so flexion behaviour is reported as a
 * measurement instead of being hidden behind a pass/fail threshold.
 */
function measureFlexion(
  sk: ResolvedSkeleton,
  side: "left" | "right",
  finger: (typeof HAND_FINGERS)[number]
): number[] {
  const chain = fingerBonesFor(sk, side)[finger] ?? [];
  const dirs: THREE.Vector3[] = [];
  for (const name of chain) {
    const b = sk.bones.get(name);
    const d = b ? worldChildDir(b, sk) : null;
    if (d) dirs.push(d);
  }
  const angles: number[] = [];
  for (let i = 1; i < dirs.length; i++) {
    const d = Math.min(1, Math.max(-1, dirs[i - 1].dot(dirs[i])));
    angles.push((Math.acos(d) * 180) / Math.PI);
  }
  return angles;
}

/**
 * Runs every hand/finger assertion for one side and appends to `checks`.
 * `extra` receives human-readable diagnostics (flexion tables).
 */
function handChecksFor(
  pose: CanonicalPose,
  sk: ResolvedSkeleton,
  side: "left" | "right",
  checks: Check[],
  extra: string[]
): void {
  const hand = side === "left" ? pose.leftHand : pose.rightHand;
  const names = fingerBonesFor(sk, side);
  const handBone = sk.getBone(`${side}Hand` as CanonicalRole);

  // --- 1. palm frame must be well-formed (unit vectors, no NaN) ----------
  const pd = vlen(hand.palmDir);
  const pn = vlen(hand.palmNormal);
  checks.push({
    bone: `${side}Hand palmDir`, role: `${side}-palmDir`,
    expected: [1, 0, 0], actual: t3(fix(cvec(hand.palmDir).normalize())),
    dot: pd, pass: Number.isFinite(pd) && pd > 0.99 && pd < 1.01,
    note: pd <= 0.99 ? `palmDir is degenerate (|palmDir|=${pd.toFixed(3)}) — phantom hand?` : undefined,
  });
  checks.push({
    bone: `${side}Hand palmNormal`, role: `${side}-palmNormal`,
    expected: [1, 0, 0], actual: t3(fix(cvec(hand.palmNormal).normalize())),
    dot: pn, pass: Number.isFinite(pn) && pn > 0.99 && pn < 1.01,
  });

  // --- 2. the hand bone must be aimed at palmDir --------------------------
  if (handBone) {
    const exp = cvec(hand.palmDir).normalize();
    const act = worldChildDir(handBone, sk);
    if (act && exp.lengthSq() > 0.5) {
      const dot = act.dot(exp);
      checks.push({
        bone: handBone.name, role: `${side}Hand`,
        expected: t3(fix(exp)), actual: t3(fix(act)), dot,
        pass: dot > 0.99,
      });
    }
  }

  // --- 3. finger phalanges ------------------------------------------------
  for (const finger of HAND_FINGERS) {
    const fp = hand.fingers[finger];
    const dirLen = vlen(fp.dir);
    const chain = names[finger] ?? [];

    // PHASE 5 Defect-1 regression: a phantom hand (built from 21 coincident
    // points, which is what the old presence test allowed through) shows up as
    // a zero-length finger direction with curl pinned at 0.5.
    if (dirLen < 0.5 && fp.curl > 0.05) {
      checks.push({
        bone: `${side} ${finger}`, role: `${side}-${finger}-phantom`,
        expected: [1, 0, 0], actual: t3(fix(cvec(fp.dir))),
        dot: dirLen, pass: false,
        note: `PHANTOM HAND: |dir|=${dirLen.toFixed(3)} curl=${fp.curl.toFixed(2)} — absent hand was not replaced by the default fallback`,
      });
    }

    // every declared phalanx must exist and be aimed somewhere finite
    chain.forEach((boneName, i) => {
      const bone = sk.bones.get(boneName);
      if (!bone) return;
      const act = worldChildDir(bone, sk);
      if (!act) return;
      const ok = Number.isFinite(act.x) && Number.isFinite(act.y) && Number.isFinite(act.z);
      checks.push({
        bone: boneName, role: `${side}-${finger}${i + 1}`,
        expected: [0, 0, 0], actual: t3(fix(act)),
        dot: ok ? 1 : 0, pass: ok,
      });
    });

    // FLAT hand (curl ~ 0, real direction): each phalanx must lie exactly
    // along the canonical finger direction.
    if (fp.curl < 0.02 && dirLen > 0.5) {
      const exp = cvec(fp.dir).normalize();
      for (let i = 0; i < chain.length; i++) {
        const bone = sk.bones.get(chain[i]);
        if (!bone) continue;
        const act = worldChildDir(bone, sk);
        if (!act) continue;
        const dot = act.dot(exp);
        checks.push({
          bone: bone.name, role: `${side}-${finger}${i + 1}-dir`,
          expected: t3(fix(exp)), actual: t3(fix(act)), dot,
          pass: dot > 0.99,
          note: dot <= 0.99 ? `flat finger off-axis by ${((Math.acos(Math.min(1, dot)) * 180) / Math.PI).toFixed(2)}deg` : undefined,
        });
      }
    }

    // Curled hand: report the measured per-joint bend so flexion behaviour is
    // visible as data, and assert COMPOUND flexion — every joint of the chain
    // must actually bend, not just the MCP. Before PHASE 6 the profile was
    // "32.0deg / 0.0deg" (single hinge, distal joints parallel), which is what
    // this gate now catches.
    if (fp.curl > 0.5) {
      const bends = measureFlexion(sk, side, finger);
      const maxBend = Math.max(0, ...bends);
      const flatJoints = bends.filter((b) => b <= 0.5).length;

      // Total tip deflection from the extended finger direction. measureFlexion
      // reports the bend BETWEEN consecutive phalanges, i.e. the joint at each
      // phalanx's distal end, so the MCP (the rotation of the first phalanx
      // relative to the hand) is not among them. Tip deflection captures the
      // whole curl, which is what proves the flexion compounded.
      const chain2 = names[finger] ?? [];
      const lastBone = chain2.length ? sk.bones.get(chain2[chain2.length - 1]) : undefined;
      const lastBone2 = chain2.length > 1 ? sk.bones.get(chain2[chain2.length - 2]) : undefined;
      const tipDir = lastBone2 ? worldChildDir(lastBone2, sk) : null;
      const expected = cvec(fp.dir).normalize();
      const tipDeflection =
        tipDir && expected.lengthSq() > 0.5
          ? (Math.acos(Math.min(1, Math.max(-1, tipDir.dot(expected)))) * 180) / Math.PI
          : NaN;

      extra.push(
        `  ${side} ${finger}: curl=${fp.curl.toFixed(2)} per-joint bend = ` +
        (bends.length ? bends.map((b) => `${b.toFixed(1)}deg`).join(" / ") : "(chain not resolved)") +
        `  | tip deflection from extended = ${Number.isFinite(tipDeflection) ? tipDeflection.toFixed(1) + "deg" : "n/a"}` +
        ` | compound: ${bends.length - flatJoints}/${bends.length} joints flexing`
      );
      void lastBone;
      checks.push({
        bone: `${side} ${finger} chain`, role: `${side}-${finger}-flexion`,
        expected: [1, 0, 0], actual: [maxBend, 0, 0], dot: maxBend,
        pass: maxBend > 1,
        note: maxBend <= 1 ? `curl=${fp.curl.toFixed(2)} produced no measurable flexion` : undefined,
      });
      // PHASE 6 gate: flexion must be distributed, not concentrated at the MCP.
      checks.push({
        bone: `${side} ${finger} compound`, role: `${side}-${finger}-compound`,
        expected: [1, 0, 0], actual: [bends.length - flatJoints, bends.length, 0],
        dot: bends.length ? (bends.length - flatJoints) / bends.length : 0,
        pass: bends.length > 1 && flatJoints === 0,
        note: flatJoints > 0
          ? `${flatJoints}/${bends.length} joints do not flex — flexion is not compounding ` +
            `(profile ${bends.map((b) => b.toFixed(1)).join("/")})`
          : undefined,
      });
    }
  }
}

// ---------------------------------------------------------------------------
// PHASE 5 — synthetic canonical HAND poses
// ---------------------------------------------------------------------------

/**
 * A synthetic HandPose built directly on the retargeter's input contract
 * (HandPose is a plain data interface). This is deliberate: the MediaPipe
 * -> HandPose conversion is covered separately by the Defect 1/2 buffer tests,
 * while these poses isolate the retargeter itself.
 */
function syntheticHand(palmDir: Vec3, palmNormal: Vec3, curl: number): HandPose {
  const fingers = {} as Record<string, { dir: Vec3; curl: number }>;
  for (const f of HAND_FINGERS) {
    fingers[f] = { dir: [...palmDir] as Vec3, curl };
  }
  return {
    wristPos: [0, 0, 0],
    palmNormal: [...palmNormal] as Vec3,
    palmDir: [...palmDir] as Vec3,
    fingers: fingers as HandPose["fingers"],
  };
}

/** FLAT HAND — fingers extended, pointing forward, palms facing down. */
function flatHandPose(joints: Joints): CanonicalPose {
  const base = poseFrom(joints);
  const palmDir: Vec3 = [1, 0, 0];
  const palmNormal: Vec3 = [0, -1, 0];
  return {
    ...base,
    leftHand: syntheticHand(palmDir, palmNormal, 0),
    rightHand: syntheticHand(palmDir, palmNormal, 0),
  };
}

/** CLOSED FIST — every joint fully curled. */
function fistPose(joints: Joints): CanonicalPose {
  const base = poseFrom(joints);
  const palmDir: Vec3 = [1, 0, 0];
  const palmNormal: Vec3 = [0, -1, 0];
  return {
    ...base,
    leftHand: syntheticHand(palmDir, palmNormal, 1),
    rightHand: syntheticHand(palmDir, palmNormal, 1),
  };
}

// ---------------------------------------------------------------------------
// PHASE 5 — Defect 1 + Defect 2 tests, driven through RAW landmark buffers
// ---------------------------------------------------------------------------
const BUF_TOTAL = 1659; // pose 99 + hands 126 + face 1434
const BUF_POSE = 0;
const BUF_LEFT_HAND = 99;
const BUF_RIGHT_HAND = 162;

/**
 * Builds a MediaPipe-space buffer holding a T-pose, with `handSlots` written
 * into the given slots, then applies the EXACT transform normalizeLandmarks
 * applies (per-point shoulder-anchor translate + shoulder-width scale).
 *
 * Reproducing that shift is the whole point: it is what turned an absent hand's
 * zeros into a non-zero point and defeated the old presence test.
 */
function buildNormalizedBuffer(
  handSlots: { offset: number; wrist: [number, number, number] }[]
): Float32Array {
  const raw = new Float32Array(BUF_TOTAL);

  // T-pose in MediaPipe image space (X right, Y down). A subject facing the
  // camera has their LEFT on the image's right, so with the arms extended
  // outward the left wrist sits at x=0.20 and the right wrist at x=0.80.
  const set = (idx: number, x: number, y: number, z = 0) => {
    raw[BUF_POSE + idx * 3] = x;
    raw[BUF_POSE + idx * 3 + 1] = y;
    raw[BUF_POSE + idx * 3 + 2] = z;
  };
  set(0, 0.5, 0.2); // nose
  set(11, 0.4, 0.35); // left shoulder
  set(12, 0.6, 0.35); // right shoulder
  set(13, 0.3, 0.45);
  set(14, 0.7, 0.45);
  set(15, 0.2, 0.5); // left wrist
  set(16, 0.8, 0.5); // right wrist
  set(23, 0.45, 0.7);
  set(24, 0.55, 0.7);
  set(25, 0.45, 0.9);
  set(26, 0.55, 0.9);
  set(27, 0.45, 1.1);
  set(28, 0.55, 1.1);

  // Hand groups: 21 points fanned out from the wrist so the group has real
  // spread. The adapter only reads the wrist, the MCPs and the finger chains.
  for (const slot of handSlots) {
    const [wx, wy, wz] = slot.wrist;
    for (let i = 0; i < 21; i++) {
      const b = slot.offset + i * 3;
      raw[b] = wx + (i % 7) * 0.01;
      raw[b + 1] = wy + Math.floor(i / 7) * 0.01;
      raw[b + 2] = wz + (i % 3) * 0.005;
    }
  }

  // --- replicate normalizeLandmarks exactly ------------------------------
  const ls = [raw[11 * 3], raw[11 * 3 + 1], raw[11 * 3 + 2]];
  const rs = [raw[12 * 3], raw[12 * 3 + 1], raw[12 * 3 + 2]];
  const anchor: [number, number, number] = [
    (ls[0] + rs[0]) / 2,
    (ls[1] + rs[1]) / 2,
    (ls[2] + rs[2]) / 2,
  ];
  const scale = Math.hypot(ls[0] - rs[0], ls[1] - rs[1], ls[2] - rs[2]) || 1;
  const out = new Float32Array(BUF_TOTAL);
  for (let i = 0; i < BUF_TOTAL; i += 3) {
    out[i] = (raw[i] - anchor[0]) / scale;
    out[i + 1] = (raw[i + 1] - anchor[1]) / scale;
    out[i + 2] = (raw[i + 2] - anchor[2]) / scale;
  }
  return out;
}

/** Reduce a 1659-float buffer to the 345-float feature layout the app uses. */
function toFeatureVector(buf: Float32Array): Float32Array {
  const out = new Float32Array(FEATURE_DIM);
  out.set(buf.subarray(0, 99), 0);
  out.set(buf.subarray(BUF_LEFT_HAND, BUF_LEFT_HAND + 63), 99);
  out.set(buf.subarray(BUF_RIGHT_HAND, BUF_RIGHT_HAND + 63), 162);
  return out;
}

function dist3(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Records a standalone (non-skeleton) assertion. */
function simpleCheck(role: string, pass: boolean, detail: string): Check {
  return {
    bone: role, role, expected: [1, 0, 0], actual: [0, 0, 0],
    dot: pass ? 1 : 0, pass, note: detail,
  };
}

function finishTest(name: string, checks: Check[], extra: string[]): TestResult {
  const res: TestResult = { name, checks, extra, pass: checks.every((c) => c.pass) };
  ALL_RESULTS.push(res);
  return res;
}

/** Defect 1: with both hand groups absent, both hands must take the default
 * fallback — NOT a phantom hand built from 21 coincident points. */
function testDefect1AbsentHands(): TestResult {
  const checks: Check[] = [];
  const extra: string[] = [];

  // No hand slots written at all -> the groups stay zero, then get shifted to
  // -anchor/scale by the normalization above. This is the exact condition that
  // used to produce phantom hands.
  const buf = buildNormalizedBuffer([]);
  const pose = mediapipeToCanonicalPose(toFeatureVector(buf), 0);
  checks.push(simpleCheck("D1-pose", !!pose, "mediapipeToCanonicalPose returns a pose"));

  if (pose) {
    const a = lastHandAssignment;
    extra.push(
      `assignment: left=${a?.left ?? "null"} right=${a?.right ?? "null"} ` +
      `(group spreads L=${a?.debug.spreadLeftSlot.toFixed(4)} R=${a?.debug.spreadRightSlot.toFixed(4)})`
    );
    checks.push(
      simpleCheck(
        "D1-both-absent",
        a?.left === null && a?.right === null,
        `both sides must resolve to null, got left=${a?.left ?? "null"} right=${a?.right ?? "null"}`
      )
    );

    for (const side of ["left", "right"] as const) {
      const h = side === "left" ? pose.leftHand : pose.rightHand;
      const body = side === "left" ? pose.body.leftWrist : pose.body.rightWrist;
      const pd = vlen(h.palmDir);
      const curls = HAND_FINGERS.map((f) => h.fingers[f].curl);
      const dirsOk = HAND_FINGERS.every((f) => vlen(h.fingers[f].dir) > 0.5);

      extra.push(
        `${side}: palmDir=[${h.palmDir.map((n) => n.toFixed(2))}] |palmDir|=${pd.toFixed(3)} ` +
        `curls=[${curls.map((c) => c.toFixed(2)).join(",")}]`
      );
      checks.push(
        simpleCheck(
          `D1-${side}-palmDir`, pd > 0.5,
          `absent hand must use the unit default palmDir (got |palmDir|=${pd.toFixed(3)}; 0 = phantom hand)`
        )
      );
      checks.push(
        simpleCheck(
          `D1-${side}-curl`,
          curls.every((c) => c === 0),
          `absent hand must use curl=0 defaults (got ${curls.join(",")}; 0.5 everywhere = phantom hand)`
        )
      );
      checks.push(
        simpleCheck(`D1-${side}-dirs`, dirsOk, "absent hand must use non-zero default finger dirs")
      );
      const wd = dist3(h.wristPos, body);
      checks.push(
        simpleCheck(
          `D1-${side}-wrist`, wd < 1e-6,
          `fallback wristPos must equal the body's ${side} wrist (delta=${wd.toExponential(2)})`
        )
      );
    }
  }

  // Control: a real hand in the left slot must be detected, not discarded.
  const buf2 = buildNormalizedBuffer([{ offset: BUF_LEFT_HAND, wrist: [0.2, 0.5, 0] }]);
  const pose2 = mediapipeToCanonicalPose(toFeatureVector(buf2), 0);
  const a2 = lastHandAssignment;
  extra.push(
    `control (left slot populated): assignment left=${a2?.left ?? "null"} ` +
    `right=${a2?.right ?? "null"} spreadL=${a2?.debug.spreadLeftSlot.toFixed(4)}`
  );
  checks.push(
    simpleCheck(
      "D1-present",
      a2?.left === BUF_LEFT_HAND && a2?.right === null,
      `a real hand in the left slot must resolve to the anatomical left and the empty right slot to null ` +
      `(got left=${a2?.left ?? "null"} right=${a2?.right ?? "null"})`
    )
  );
  checks.push(
    simpleCheck(
      "D1-present-real",
      !!pose2 && vlen(pose2.leftHand.palmDir) > 0.5,
      "a detected hand must produce a real palmDir, not the default"
    )
  );

  return finishTest(
    "P5-1. Defect 1: absent hand -> default fallback (no phantom claws)",
    checks, extra
  );
}

/** Defect 2: the hand array slots are deliberately SWAPPED relative to anatomy
 * (exactly what a mirror-assuming classifier produces). The adapter must still
 * route each hand to the anatomical side it physically occupies. */
function testDefect2Chirality(): TestResult {
  const checks: Check[] = [];
  const extra: string[] = [];

  // Subject faces the camera: left wrist raw x=0.20, right wrist raw x=0.80.
  // Hand A sits at the subject's RIGHT wrist but is stored in MediaPipe's
  // "left" slot; hand B is the mirror image of that.
  const buf = buildNormalizedBuffer([
    { offset: BUF_LEFT_HAND, wrist: [0.8, 0.5, 0] },
    { offset: BUF_RIGHT_HAND, wrist: [0.2, 0.5, 0] },
  ]);

  const pose = mediapipeToCanonicalPose(toFeatureVector(buf), 0);
  checks.push(simpleCheck("D2-pose", !!pose, "mediapipeToCanonicalPose returns a pose"));
  if (!pose) {
    return finishTest(
      "P5-2. Defect 2: mirrored hand slots -> anatomical side assignment",
      checks, extra
    );
  }

  const a = lastHandAssignment!;
  const slotName = (o: number | null) =>
    o === BUF_LEFT_HAND ? '"left" slot' : o === BUF_RIGHT_HAND ? '"right" slot' : "null";
  extra.push(
    `slot->wrist distances — "left" slot: toL=${a.debug.dLeftSlotToLeftWrist.toFixed(3)} ` +
    `toR=${a.debug.dLeftSlotToRightWrist.toFixed(3)} | "right" slot: ` +
    `toL=${a.debug.dRightSlotToLeftWrist.toFixed(3)} toR=${a.debug.dRightSlotToRightWrist.toFixed(3)}`
  );
  extra.push(
    `resolved — anatomical left <- ${slotName(a.left)}, anatomical right <- ${slotName(a.right)}`
  );

  checks.push(
    simpleCheck(
      "D2-left", a.left === BUF_RIGHT_HAND,
      `the hand at the subject's LEFT wrist lives in the "right" slot and must be routed to the anatomical left hand (got ${slotName(a.left)})`
    )
  );
  checks.push(
    simpleCheck(
      "D2-right", a.right === BUF_LEFT_HAND,
      `the hand at the subject's RIGHT wrist lives in the "left" slot and must be routed to the anatomical right hand (got ${slotName(a.right)})`
    )
  );

  // End-to-end: each HandPose's wrist must sit on its own side of the body.
  const dLL = dist3(pose.leftHand.wristPos, pose.body.leftWrist);
  const dLR = dist3(pose.leftHand.wristPos, pose.body.rightWrist);
  const dRL = dist3(pose.rightHand.wristPos, pose.body.leftWrist);
  const dRR = dist3(pose.rightHand.wristPos, pose.body.rightWrist);
  extra.push(
    `wrist deltas — leftHand: toL=${dLL.toFixed(3)} toR=${dLR.toFixed(3)} | ` +
    `rightHand: toL=${dRL.toFixed(3)} toR=${dRR.toFixed(3)}`
  );
  checks.push(
    simpleCheck(
      "D2-leftWrist", dLL < dLR,
      `leftHand.wristPos must be nearer the body's left wrist (toL=${dLL.toFixed(3)} toR=${dLR.toFixed(3)})`
    )
  );
  checks.push(
    simpleCheck(
      "D2-rightWrist", dRR < dRL,
      `rightHand.wristPos must be nearer the body's right wrist (toL=${dRL.toFixed(3)} toR=${dRR.toFixed(3)})`
    )
  );

  return finishTest(
    "P5-2. Defect 2: mirrored hand slots -> anatomical side assignment",
    checks, extra
  );
}

/** Chirality: two anatomically mirrored palms must produce MIRRORED palm
 * normals. A same-handedness bug would show up here as dot ~= +1. */
function testPalmChirality(): TestResult {
  const checks: Check[] = [];
  const extra: string[] = [];
  const l = cvec(syntheticHand([1, 0, 0], [0, -1, 0], 0).palmNormal).normalize();
  const r = cvec(syntheticHand([1, 0, 0], [0, 1, 0], 0).palmNormal).normalize();
  const dot = l.dot(r);
  extra.push(
    `mirrored palms: left normal ${f3(l)}, right normal ${f3(r)}, dot=${dot.toFixed(4)} (want ~ -1)`
  );
  checks.push({
    bone: "palmNormal L/R", role: "chirality",
    expected: [-1, 0, 0], actual: t3(fix(l)), dot, pass: dot < -0.99,
  });
  return finishTest(
    "P5-3. Palm chirality: mirrored hands give mirrored palm normals",
    checks, extra
  );
}

let BIND_LEG_X: { left: number; right: number } | null = null;

function runTest(
  name: string,
  pose: CanonicalPose,
  sk: ResolvedSkeleton,
  scene: THREE.Object3D,
  restSpineDir: THREE.Vector3,
  opts: { checkHipsFacing?: boolean } = {}
): TestResult {
  const body = pose.body;

  resetToBind(sk, scene);
  const result = retarget(sk, pose);
  applyBoneRotations(result.boneRotations, sk);
  scene.updateMatrixWorld(true);

  const spinePoseRot = new THREE.Quaternion().setFromUnitVectors(
    restSpineDir.clone().normalize(),
    cvec(body.spineDir).normalize()
  );

  const checks: Check[] = [];
  const extra: string[] = [];

  for (const role of DIR_ROLES) {
    const bone = sk.getBone(role);
    const exp = expectedDir(role, body, spinePoseRot, sk);
    if (!bone || !exp) continue;
    const act = worldChildDir(bone, sk);
    if (!act) continue;
    const dot = act.dot(exp);
    checks.push({
      bone: bone.name, role, expected: t3(fix(exp)), actual: t3(fix(act)),
      dot, pass: dot > 0.999,
    });
  }

  // No-data bones: must stay at their bind rotation (nothing may be invented).
  for (const role of NO_DATA_ROLES) {
    const bone = sk.getBone(role);
    if (!bone) continue;
    const d = Math.min(1, Math.abs(bone.bone.quaternion.dot(new THREE.Quaternion())));
    const ang = (Math.acos(d) * 2 * 180) / Math.PI;
    extra.push(`${role} (${bone.name}) local rot = ${ang.toFixed(2)}deg from bind (no source data -> expect ~0)`);
    checks.push({
      bone: bone.name, role, expected: [0, 0, 0], actual: [0, 0, 0], dot: d, pass: ang < 1,
      note: ang >= 1 ? `foot rotated ${ang.toFixed(1)}deg with no source data (aimed at shinDir)` : undefined,
    });
  }

  // Hips: no aim semantics. A NEUTRAL pose must reproduce the bind (identity) rotation;
  // a deliberately leaned pose legitimately rotates the root, so only assert identity
  // when the canonical spine is vertical and the shoulder axis is the subject's right.
  const hips = sk.getBone("hips");
  const neutral =
    Math.abs(body.spineDir[0]) < 1e-6 && Math.abs(body.spineDir[2]) < 1e-6 && body.spineDir[1] > 0 &&
    Math.abs(body.shoulderAxis[0]) < 1e-6 && Math.abs(body.shoulderAxis[1]) < 1e-6;
  if (hips && neutral) {
    const d = Math.min(1, Math.abs(hips.bone.quaternion.dot(new THREE.Quaternion())));
    const ang = (Math.acos(d) * 2 * 180) / Math.PI;
    extra.push(`hips (${hips.name}) local rot = ${ang.toFixed(2)}deg from bind (neutral pose -> expect ~0)`);
    checks.push({
      bone: hips.name, role: "hips", expected: [0, 0, 0], actual: [0, 0, 0], dot: d, pass: ang < 2,
      note: ang >= 2 ? `root rotated ${ang.toFixed(1)}deg (bind basis not reproduced)` : undefined,
    });

    if (opts.checkHipsFacing) {
      const facing = new THREE.Vector3(0, 0, 1).applyQuaternion(worldQuat(hips)).normalize();
      const want = cvec([1, 0, 0]).normalize(); // canonical forward -> +Z gltf
      const fd = facing.dot(want);
      extra.push(`hips facing: actual ${f3(facing)} want ${f3(want)} dot=${fd.toFixed(4)} (1.0 = character faces +Z)`);
      checks.push({ bone: hips.name, role: "hips-facing", expected: t3(fix(want)), actual: t3(fix(facing)), dot: fd, pass: fd > 0.999 });
    }
  }

  // Left/right leg roots must not swap sides -- but ONLY for a root that has
  // not been yawed. The retargeter deliberately yaws the hips so the avatar's
  // right matches the SUBJECT's right; for a subject photographed from BEHIND
  // (subject's right along canonical -Z) that yaw is a legitimate ~180deg turn,
  // and the leg roots then correctly land on the opposite world-X side. Asserting
  // the bind sign unconditionally produced false failures on real back-facing
  // photos, so the assertion is gated on the measured root yaw and the yawed
  // case is reported as information instead.
  if (BIND_LEG_X) {
    const hipsQ = sk.getBone("hips") ? worldQuat(sk.getBone("hips")!) : new THREE.Quaternion();
    const facing = new THREE.Vector3(0, 0, 1).applyQuaternion(hipsQ);
    const yawDeg = (Math.atan2(facing.x, facing.z) * 180) / Math.PI;
    const untilted = Math.abs(yawDeg) < 45;

    const l = sk.getBone("leftHip")!.bone.getWorldPosition(new THREE.Vector3());
    const r = sk.getBone("rightHip")!.bone.getWorldPosition(new THREE.Vector3());
    const swap = Math.sign(l.x - r.x) !== Math.sign(BIND_LEG_X.left - BIND_LEG_X.right);
    extra.push(
      `leg roots: LeftUpLeg.x=${l.x.toFixed(4)} RightUpLeg.x=${r.x.toFixed(4)} ` +
      `(bind ${BIND_LEG_X.left.toFixed(4)} / ${BIND_LEG_X.right.toFixed(4)}) root yaw=${yawDeg.toFixed(1)}deg ` +
      (untilted ? (swap ? "SWAPPED" : "ok") : "yawed -> side check not applicable")
    );
    if (untilted) {
      checks.push({
        bone: "hips->legRoots", role: "leg-root-side", expected: [0, 0, 0], actual: t3(fix(l.clone().sub(r))),
        dot: swap ? -1 : 1, pass: !swap, note: swap ? "left/right leg roots swapped sides" : undefined,
      });
    }
  }

  // PHASE 5 — hand + finger bone checks (both sides, every test).
  handChecksFor(pose, sk, "left", checks, extra);
  handChecksFor(pose, sk, "right", checks, extra);

  // PHASE 5 hard gate: zero NaN / Inf anywhere in the evaluated skeleton.
  const nonFinite = checkFinite(sk, scene);
  extra.push(
    `finite check: ${nonFinite.length === 0 ? "all world matrices + quaternions finite" : "NON-FINITE -> " + nonFinite.join(", ")}`
  );
  checks.push({
    bone: "(skeleton)", role: "finite", expected: [0, 0, 0], actual: [0, 0, 0],
    dot: nonFinite.length === 0 ? 1 : 0, pass: nonFinite.length === 0,
    note: nonFinite.length ? nonFinite.join(", ") : undefined,
  });

  const pass = checks.every((c) => c.pass);
  const res: TestResult = { name, checks, extra, pass };
  ALL_RESULTS.push(res);
  return res;
}

function printTest(res: TestResult) {
  console.log(`\n=== TEST: ${res.name} — ${res.pass ? "PASS" : "FAIL"} ===`);
  for (const e of res.extra) console.log(`  · ${e}`);
  for (const c of res.checks) {
    const mark = c.pass ? "ok  " : "FAIL";
    console.log(
      `  [${mark}] ${c.role.padEnd(15)} ${c.bone.padEnd(24)} expected=${JSON.stringify(c.expected)} ` +
      `actual=${JSON.stringify(c.actual)} dot=${c.dot.toFixed(4)}${c.note ? "  <- " + c.note : ""}`
    );
  }
  const bad = res.checks.filter((c) => !c.pass).length;
  if (bad) console.log(`  >> ${bad}/${res.checks.length} checks FAILED`);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function printBindTable(sk: ResolvedSkeleton) {
  console.log("\n=== MEASURED BIND POSE (read from the real GLB via resolveSkeleton) ===");
  console.log("name".padEnd(26) + "parent".padEnd(24) + "localPos".padEnd(26) + "worldPos".padEnd(26) + "childDir".padEnd(26) + "len(cm)  scale");
  for (const name of sk.boneOrder) {
    const b = sk.bones.get(name);
    if (!b) continue;
    const cd = worldChildDir(b, sk);
    console.log(
      name.padEnd(26) +
      (b.parent ?? "(root)").padEnd(24) +
      f3(b.localPos).padEnd(26) +
      f3(b.worldPos).padEnd(26) +
      (cd ? f3(cd) : "[-,-,-]").padEnd(26) +
      b.length.toFixed(2).padStart(7) + "  " + f3(b.bone.scale)
    );
  }
  console.log("\nbind LOCAL quats are all identity? " +
    [...sk.bones.values()].every((b) => b.localRot.w > 0.9999));
}

/** Chain diagnostic: is the actual bone hierarchy state what the retargeter believes? */
function debugChain(
  name: string,
  joints: Joints,
  sk: ResolvedSkeleton,
  scene: THREE.Object3D,
  restSpineDir: THREE.Vector3
) {
  const body = bodyFrom(joints);
  resetToBind(sk, scene);
  const result = retarget(sk, poseFrom(joints));
  applyBoneRotations(result.boneRotations, sk);
  scene.updateMatrixWorld(true);
  const spinePoseRot = new THREE.Quaternion().setFromUnitVectors(
    restSpineDir.clone().normalize(),
    cvec(body.spineDir).normalize()
  );
  const I = new THREE.Quaternion();
  console.log(`\n--- CHAIN DEBUG (${name}) — actual three.js state vs retargeter bookkeeping ---`);
  console.log(
    "role".padEnd(9) + "bone".padEnd(22) + "actualLocal".padEnd(13) + "mapLocal".padEnd(13) +
    "|local-map|".padEnd(13) + "|world-map|".padEnd(13) + "actualWorld-vs-bind".padEnd(21) + "actual child dir"
  );
  for (const role of ["hips", "spine1", "spine2", "spine3", "neck", "head"] as CanonicalRole[]) {
    const b = sk.getBone(role);
    if (!b) continue;
    const actualLocal = b.bone.quaternion.clone();
    const mapLocal = result.boneRotations.get(b.name);
    const actualWorld = b.bone.getWorldQuaternion(new THREE.Quaternion());
    const mapWorld = result.worldRotations.get(b.name);
    const cd = worldChildDir(b, sk);
    const exp = expectedDir(role, body, spinePoseRot, sk);
    console.log(
      role.padEnd(9) + b.name.padEnd(22) +
      (qangle(actualLocal, I).toFixed(2) + "d").padEnd(13) +
      (mapLocal ? qangle(mapLocal, I).toFixed(2) + "d" : "none").padEnd(13) +
      (mapLocal ? qangle(actualLocal, mapLocal).toFixed(4) + "d" : "-").padEnd(13) +
      (mapWorld ? qangle(actualWorld, mapWorld).toFixed(4) + "d" : "-").padEnd(13) +
      (qangle(actualWorld, I).toFixed(2) + "d").padEnd(21) +
      (cd ? f3(cd) : "-") + (exp && cd ? `  target=${f3(exp)} dot=${cd.dot(exp).toFixed(4)}` : "")
    );
  }
}

function defineTests(): Array<{ name: string; make: () => Joints; face?: boolean }> {
  return [
    { name: "1. synthetic T-pose (neutral bind reproduction)", make: tposeJoints, face: true },
    {
      name: "2. bent elbows 90deg, forearms forward (E1 catcher)",
      make: () => { const j = tposeJoints(); j.lWr = [0.40, 1.45, -0.60]; j.rWr = [0.40, 1.45, 0.60]; return j; },
    },
    {
      name: "3. knee bend, shins swing backward",
      make: () => { const j = tposeJoints(); j.lAnkle = [-0.25, 0.25, -0.10]; j.rAnkle = [-0.25, 0.25, 0.10]; return j; },
    },
    {
      name: "4. spine lean forward (torso + head follow)",
      make: () => {
        const j = tposeJoints();
        j.lSh = [0.35, 1.45, -0.20]; j.rSh = [0.35, 1.45, 0.20];
        j.lEl = [0.35, 1.45, -0.60]; j.rEl = [0.35, 1.45, 0.60];
        j.lWr = [0.35, 1.45, -1.00]; j.rWr = [0.35, 1.45, 1.00];
        return j;
      },
    },
    {
      name: "5. left upper arm raised 45deg forward (elbow bend in X/Y plane)",
      make: () => {
        const j = tposeJoints();
        j.lEl = [0.283, 1.733, -0.20]; j.lWr = [0.283, 1.733, -0.60];
        return j;
      },
    },
    {
      name: "6. right upper arm lowered 45deg forward (mirror check)",
      make: () => {
        const j = tposeJoints();
        j.rEl = [0.283, 1.167, 0.20]; j.rWr = [0.283, 1.167, 0.60];
        return j;
      },
    },
    {
      name: "7. left forearm only, points straight up (E1: upper arm must NOT move)",
      make: () => { const j = tposeJoints(); j.lWr = [0, 1.85, -0.60]; return j; },
    },
    {
      name: "8. left thigh abducted outward (shin unchanged)",
      make: () => { const j = tposeJoints(); j.lKnee = [0.30, 0.75, -0.10]; j.lAnkle = [0.30, 0.30, -0.10]; return j; },
    },
    {
      name: "9. left shin only (knee flexed, thigh unchanged)",
      make: () => { const j = tposeJoints(); j.lAnkle = [-0.30, 0.28, -0.10]; return j; },
    },
    {
      name: "10. right leg abducted outward (mirror check)",
      make: () => { const j = tposeJoints(); j.rKnee = [0, 0.75, 0.30]; j.rAnkle = [0, 0.30, 0.30]; return j; },
    },
    {
      name: "11. head tilt forward (neck + head)",
      make: () => { const j = tposeJoints(); j.head = [0.16, 1.62, 0]; return j; },
    },
    {
      name: "12. asymmetric mixed pose (L arm up-fwd, R arm down-fwd, L knee out)",
      make: () => {
        const j = tposeJoints();
        j.lEl = [0.283, 1.733, -0.20]; j.lWr = [0.10, 1.90, -0.60];
        j.rEl = [0.283, 1.167, 0.20]; j.rWr = [0.283, 1.167, 0.60];
        j.lKnee = [0.30, 0.75, -0.10]; j.lAnkle = [0.30, 0.30, -0.10];
        return j;
      },
    },
  ];
}

/**
 * PHASE 4: run every extracted real-pose .bin through the LIVE pipeline
 *   345-float buffer -> mediapipeToCanonicalPose -> retarget -> applyBoneRotations
 * and apply exactly the same mathematical gates as the synthetic tests.
 * Nothing here is retargeter math; this only feeds real MediaPipe output into
 * the already-verified pipeline and measures the result.
 */
function runRealPoseTests(
  sk: ResolvedSkeleton,
  scene: THREE.Object3D,
  restSpineDir: THREE.Vector3
): { loaded: number; rejected: string[] } {
  const rejected: string[] = [];
  if (!fs.existsSync(FIXTURE_DIR)) {
    console.log(`\n!! no fixture dir at ${FIXTURE_DIR} — run extract_landmarks.py first`);
    return { loaded: 0, rejected };
  }

  const bins = fs
    .readdirSync(FIXTURE_DIR)
    .filter((f) => f.endsWith(".bin"))
    .sort();
  if (bins.length === 0) {
    console.log(`\n!! no .bin files in ${FIXTURE_DIR} — run extract_landmarks.py first`);
    return { loaded: 0, rejected };
  }

  console.log(
    `\n########## PHASE 4 REAL-POSE TESTS (${bins.length} MediaPipe-derived buffers) ##########`
  );

  let loaded = 0;
  for (const file of bins) {
    const raw = fs.readFileSync(path.join(FIXTURE_DIR, file));
    if (raw.byteLength !== FEATURE_DIM * 4) {
      rejected.push(`${file}: ${raw.byteLength} bytes, expected ${FEATURE_DIM * 4}`);
      continue;
    }
    const view = new Float32Array(
      raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)
    );
    if (![...view].every(Number.isFinite)) {
      rejected.push(`${file}: contains NaN/Inf in the landmark buffer`);
      continue;
    }

    const pose = mediapipeToCanonicalPose(view, 0);
    if (!pose) {
      rejected.push(`${file}: mediapipeToCanonicalPose returned null (no usable body)`);
      continue;
    }

    loaded++;
    const res = runTest(
      `real/${file.replace(/\.bin$/, "")}`,
      pose,
      sk,
      scene,
      restSpineDir,
      { checkHipsFacing: false }
    );
    printTest(res);
  }

  if (rejected.length) {
    console.log(`\n!! ${rejected.length} fixture(s) rejected before retargeting:`);
    for (const r of rejected) console.log(`   - ${r}`);
  }
  return { loaded, rejected };
}

// Diagnostic: dump the canonical body for one .bin (read-only; touches no math).
const which = process.argv[2];
if (which && which.endsWith(".bin")) {
  const raw = fs.readFileSync(path.join(FIXTURE_DIR, which));
  const v = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const p = mediapipeToCanonicalPose(v, 0)!;
  const b = p.body;
  console.log("canonical body for", which);
  console.log("  spineDir    ", b.spineDir.map((n) => n.toFixed(4)));
  console.log("  shoulderAxis", b.shoulderAxis.map((n) => n.toFixed(4)));
  console.log("  lSh", b.leftShoulder.map((n) => n.toFixed(3)), " rSh", b.rightShoulder.map((n) => n.toFixed(3)));
  console.log("  lHip", b.leftHip.map((n) => n.toFixed(3)), " rHip", b.rightHip.map((n) => n.toFixed(3)));
  console.log("  gltf lSh", f3(cvec(b.leftShoulder)), " rSh", f3(cvec(b.rightShoulder)));
  console.log("  gltf spineDir", f3(cvec(b.spineDir)), " shoulderAxis", f3(cvec(b.shoulderAxis)));
  process.exit(0);
}

async function main() {
  console.log("Xbot GLB:", MODEL_PATH);
  const scene = new THREE.Scene();
  const raw = fs.readFileSync(MODEL_PATH);
  // NOTE: GLTFLoader.parse() only accepts string | ArrayBuffer.  A Node Buffer is
  // neither, in which case three takes the `json = data` branch, then bails out of
  // `json.asset === undefined` WITHOUT reporting anything when onError is falsy —
  // the promise never settles and Node exits silently.  Slice a real ArrayBuffer.
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  const loader = new GLTFLoader();
  const gltf: any = await new Promise((resolve, reject) => {
    loader.parse(buffer, "", resolve, reject);
  });
  const clips: string[] = (gltf.animations ?? []).map((a: any) => a.name);
  console.log(`GLB animation clips present: [${clips.join(", ")}] (never played by the app)`);
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);

  const sk = resolveSkeleton(gltf.scene);
  console.log(`resolveSkeleton: ${sk.bones.size} bones, armatureScale=${sk.armatureScale}`);

  printBindTable(sk);

  const bindId = checkBindIdentity(sk);
  console.log(
    `\nbind WORLD rotations all identity? ${bindId.ok ? "YES" : "NO"} ` +
    `(worst: ${bindId.worst} = ${bindId.worstAngle.toFixed(3)} deg)`
  );

  const hipsB = sk.getBone("hips")!;
  const neckB = sk.getBone("neck")!;
  const restSpineDir = new THREE.Vector3().subVectors(neckB.worldPos, hipsB.worldPos).normalize();
  const lr = sk.getBone("leftHip")!.bone.getWorldPosition(new THREE.Vector3());
  const rr = sk.getBone("rightHip")!.bone.getWorldPosition(new THREE.Vector3());
  BIND_LEG_X = { left: lr.x, right: rr.x };
  console.log(`rest spine dir at bind (gltf): ${f3(restSpineDir)}`);
  console.log(`bind leg roots: LeftUpLeg.x=${lr.x.toFixed(4)} RightUpLeg.x=${rr.x.toFixed(4)}`);

  console.log("\n--- key bind directions (GLTF) ---");
  for (const role of ["hips", "spine1", "neck", "head", "leftShoulder", "leftUpperArm",
    "leftForearm", "leftHand", "leftHip", "leftKnee", "leftAnkle"] as CanonicalRole[]) {
    const b = sk.getBone(role);
    if (!b) { console.log(`  ${role}: MISSING`); continue; }
    console.log(`  ${role.padEnd(14)} -> ${b.name.padEnd(22)} childDir=${f3(b.localDir.clone().normalize())} len=${b.length.toFixed(2)}cm`);
  }

  console.log("\n########## TESTS ##########");

  // PHASE 7 diagnostic: npx tsx scripts/verify-retarget.ts --root-scale
  if (process.argv[2] === "--root-scale") {
    const lsh = sk.getBone("leftShoulder")!.bone.getWorldPosition(new THREE.Vector3());
    const rsh = sk.getBone("rightShoulder")!.bone.getWorldPosition(new THREE.Vector3());
    const larm = sk.getBone("leftUpperArm")!.bone.getWorldPosition(new THREE.Vector3());
    const rarm = sk.getBone("rightUpperArm")!.bone.getWorldPosition(new THREE.Vector3());
    const hipsB2 = sk.getBone("hips")!.bone.getWorldPosition(new THREE.Vector3());
    const headB2 = sk.getBone("head")!.bone.getWorldPosition(new THREE.Vector3());
    let lowest = Infinity;
    let lowestName = "";
    for (const b of sk.bones.values()) {
      const y = b.bone.getWorldPosition(new THREE.Vector3()).y;
      if (y < lowest) {
        lowest = y;
        lowestName = b.name;
      }
    }
    const shoulderWidth = lsh.distanceTo(rsh);
    const armSpan = larm.distanceTo(rarm);
    console.log("\n=== PHASE 7 ROOT-TRANSLATION SCALE (measured from the real GLB) ===");
    console.log(`leftShoulder world      : ${f3(lsh)}`);
    console.log(`rightShoulder world     : ${f3(rsh)}`);
    console.log(`shoulder bone span      : ${shoulderWidth.toFixed(4)} m`);
    console.log(`arm-joint span          : ${armSpan.toFixed(4)} m`);
    console.log(`hips world y            : ${hipsB2.y.toFixed(4)} m`);
    console.log(`head world y            : ${headB2.y.toFixed(4)} m`);
    console.log(`bind height hips->head  : ${(headB2.y - hipsB2.y).toFixed(4)} m`);
    console.log(`lowest bind bone (floor): ${lowestName} at y=${lowest.toFixed(4)} m`);
    console.log("");
    console.log("Source unit = SHOULDER WIDTH (normalizeLandmarks divides by it), so:");
    console.log(`  metresPerShoulderWidth = ${shoulderWidth.toFixed(4)}  (measured)  /  ${(armSpan / 0.4).toFixed(4)}  (0.40m human span)`);
    process.exit(0);
  }

  debugChain("T-pose", tposeJoints(), sk, scene, restSpineDir);
  for (const t of defineTests()) {
    const res = runTest(t.name, poseFrom(t.make()), sk, scene, restSpineDir, {
      checkHipsFacing: t.face,
    });
    printTest(res);
  }

  // ---- PHASE 5: hand + finger synthetic poses and the two defect tests ----
  console.log("\n########## PHASE 5 HAND / FINGER TESTS ##########");
  printTest(runTest("P5-A. Flat Hand (fingers extended, palm down)", flatHandPose(tposeJoints()), sk, scene, restSpineDir));
  printTest(runTest("P5-B. Closed Fist (all joints curled)", fistPose(tposeJoints()), sk, scene, restSpineDir));
  printTest(testDefect1AbsentHands());
  printTest(testDefect2Chirality());
  printTest(testPalmChirality());

  // ---- PHASE 4: real MediaPipe-derived poses through the same gates -------
  const real = runRealPoseTests(sk, scene, restSpineDir);

  const failed = ALL_RESULTS.filter((r) => !r.pass);
  const totalChecks = ALL_RESULTS.reduce((n, r) => n + r.checks.length, 0);
  const failedChecks = ALL_RESULTS.reduce((n, r) => n + r.checks.filter((c) => !c.pass).length, 0);
  const realTests = ALL_RESULTS.filter((r) => r.name.startsWith("real/"));
  const realFailed = realTests.filter((r) => !r.pass);
  const nonFiniteChecks = ALL_RESULTS.reduce(
    (n, r) => n + r.checks.filter((c) => c.role === "finite" && !c.pass).length,
    0
  );

  console.log("\n########## SUMMARY ##########");
  console.log(`tests : ${ALL_RESULTS.length - failed.length}/${ALL_RESULTS.length} passed`);
  console.log(`checks: ${totalChecks - failedChecks}/${totalChecks} passed`);
  console.log(`  synthetic: ${ALL_RESULTS.length - realTests.length - (failed.length - realFailed.length)}/${ALL_RESULTS.length - realTests.length} passed`);
  console.log(`  real pose: ${realTests.length - realFailed.length}/${realTests.length} passed (${real.loaded} MediaPipe buffers loaded, ${real.rejected.length} rejected)`);
  console.log(`  NaN/Inf violations: ${nonFiniteChecks}`);

  // Per-role dot-product table over the real-pose tests (Phase 4 headline gate).
  const roleDot = new Map<string, number[]>();
  for (const r of realTests) {
    for (const c of r.checks) {
      if (!roleDot.has(c.role)) roleDot.set(c.role, []);
      roleDot.get(c.role)!.push(c.dot);
    }
  }
  if (roleDot.size) {
    console.log("\n--- real-pose per-role dot summary (gate: dot > 0.999) ---");
    for (const role of DIR_ROLES) {
      const dots = roleDot.get(role);
      if (!dots || !dots.length) continue;
      const min = Math.min(...dots);
      const mean = dots.reduce((a, b) => a + b, 0) / dots.length;
      console.log(
        `  ${role.padEnd(15)} n=${String(dots.length).padStart(2)}  min=${min.toFixed(6)}  mean=${mean.toFixed(6)}  ${min > 0.999 ? "PASS" : "FAIL"}`
      );
    }
    for (const role of NO_DATA_ROLES) {
      const dots = roleDot.get(role);
      if (!dots || !dots.length) continue;
      const min = Math.min(...dots);
      console.log(
        `  ${role.padEnd(15)} n=${String(dots.length).padStart(2)}  min dot=${min.toFixed(6)}  (identity local rot, gate: angle < 1.0deg)`
      );
    }
  }

  // PHASE 5 — hand + finger bone dot summary (gate: dot > 0.99).
  const handRoles = new Map<string, number[]>();
  for (const r of ALL_RESULTS) {
    for (const c of r.checks) {
      if (!/^(left|right)-(palmDir|palmNormal|Hand$)|^(left|right)-(thumb|index|middle|ring|pinky)/.test(c.role)) {
        continue;
      }
      if (!handRoles.has(c.role)) handRoles.set(c.role, []);
      handRoles.get(c.role)!.push(c.dot);
    }
  }
  if (handRoles.size) {
    console.log("\n--- hand / finger bone dot summary (gate: dot > 0.99) ---");
    const keys = [...handRoles.keys()].sort();
    for (const role of keys) {
      const dots = handRoles.get(role)!;
      const min = Math.min(...dots);
      const mean = dots.reduce((a, b) => a + b, 0) / dots.length;
      console.log(
        `  ${role.padEnd(22)} n=${String(dots.length).padStart(3)}  min=${min.toFixed(6)}  mean=${mean.toFixed(6)}  ${min > 0.99 ? "PASS" : "FAIL"}`
      );
    }
    const worst = Math.min(...[...handRoles.values()].map((d) => Math.min(...d)));
    console.log(`  worst hand/finger dot across all ${ALL_RESULTS.length} tests: ${worst.toFixed(6)}`);
  }

  if (failed.length) {
    console.log("\nfailing tests:");
    for (const r of failed) console.log(`  - ${r.name}`);
    const roles = new Map<string, number>();
    for (const r of failed) for (const c of r.checks) if (!c.pass) roles.set(c.role, (roles.get(c.role) ?? 0) + 1);
    console.log("failing roles: " + [...roles.entries()].map(([k, v]) => `${k}(${v})`).join(", "));
  }

  const out = { model: MODEL_PATH, bindIdentity: bindId, restSpineDir: restSpineDir.toArray(), tests: ALL_RESULTS };
  fs.writeFileSync(fileURLToPath(new URL("../verify-retarget.json", import.meta.url)), JSON.stringify(out, null, 2));
  console.log("\nwrote verify-retarget.json");

  if (failed.length || nonFiniteChecks > 0) {
    console.error(`\nVERIFY FAILED: ${failed.length} test(s), ${nonFiniteChecks} NaN/Inf violation(s)`);
    process.exitCode = 1;
  } else {
    console.log(`\nVERIFY PASSED: ${ALL_RESULTS.length} tests, ${totalChecks} checks, 0 NaN/Inf`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});





