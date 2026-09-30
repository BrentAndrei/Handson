/**
 * Body retargeter: maps canonical pose body joints to avatar bone rotations.
 *
 * Algorithm per bone chain (arm, leg, spine, head):
 *   1. Compute target world-space direction from canonical pose
 *   2. Use boneResolver.bindDir as the rest direction
 *   3. dirQuat = rotationBetween(bindDir, targetDir) — shortest arc aim
 *   4. Extract twist using a secondary "side" landmark (elbow/knee plane)
 *   5. worldRot = dirQuat * twistQuat
 *   6. localRot = parentWorldRot⁻¹ * worldRot
 *
 * This replaces the broken quaternion-composition in the old retarget.ts
 * where `dirQuat * palmQuat` was used instead of `dirQuat * twistQuat`
 * and the twist was never properly extracted.
 */
import {
  Matrix4,
  Quaternion,
  Vector3,
} from "three";
import {
  BodyPose,
  Vec3,
  rotationBetween,
  CANONICAL_UP,
} from "./canonicalPose";
import {
  ResolvedSkeleton,
  ResolvedBone,
  CanonicalRole,
} from "./boneResolver";
import { canonicalToGltf, Coords } from "./coordinateSystem";

/**
 * Convert a canonical Vec3 → Three.js Vector3 in GLTF space, writing into `out`.
 *
 * PHASE 11: the original returned `new Vector3(...)`, and this is called ~20+
 * times per retarget (hips, spine, shoulder axis, every limb joint), so it was
 * one of the largest single sources of garbage. `out` is always supplied by the
 * caller now, so no allocation happens here at all. `canonicalToGltf` is a pure
 * tuple transform and is unchanged, so the numbers are bit-identical.
 */
const cvec = (v: Vec3, out: Vector3): Vector3 => {
  const g = canonicalToGltf(v as Coords);
  return out.set(g[0], g[1], g[2]);
};

const VEC_POOL: Vector3[] = [];
const QUAT_POOL: Quaternion[] = [];

/**
 * PHASE 11 — module-scoped scratch, shared by retargetBody / retargetArm /
 * retargetLeg (all three are siblings, so function-local scratch could not be
 * shared, and hoisting is what makes `cvec` allocation-free).
 *
 * WHY THIS IS SAFE (the cross-contamination question):
 *  - `retargetBody` calls `retargetArm` and `retargetLeg` SEQUENTIALLY, never
 *    re-entrantly, and never from inside a partially-consumed region of
 *    itself. The arm/leg helpers use `_hipPos`/`_kneePos`/`_anklePos` and
 *    `_shoulderPos`/`_elbowPos`/`_wristPos`; `retargetBody` uses only
 *    `_hipsPos`/`_spineDir`/`_shoulderAxis`/`_headDir`/`_mat`/`_spinePoseRot`.
 *    The two groups do not overlap, so a call into one can never clobber a
 *    value the other still needs.
 *  - Within a function, every slot is filled by `cvec()` and consumed on the
 *    same statement or the next one. No slot is held across a call that could
 *    refill it.
 *  - Each of the three joint slots is live at the same time (hip/knee/ankle),
 *    which is precisely why they are three separate objects and not one reused
 *    buffer.
 */
const _hipsPos = new Vector3();
const _spineDir = new Vector3();
const _shoulderAxis = new Vector3();
const _headDir = new Vector3();
const _shoulderPos = new Vector3();
const _elbowPos = new Vector3();
const _wristPos = new Vector3();
const _hipPos = new Vector3();
const _kneePos = new Vector3();
const _anklePos = new Vector3();
const _mat = new Matrix4();
const _spinePoseRot = new Quaternion();
// Dedicated identity scratch for the `worldRotations.get(parent) ?? <identity>`
// fallback. It MUST NOT reuse _spinePoseRot: spinePoseRot is still live at that
// point in the neck/head blocks, and sharing the slot would zero it.
const _identQ = new Quaternion();
// PHASE 11: scratch for retargetSingleBone's `bindDir`. See the ordering
// argument at the use site: it is consumed before any nested call can refill it.
const _bBindDir = new Vector3();
// PHASE 11: constant axes. These appear inside the degenerate-plane fallback
// branches of retargetSingleBone / retargetArm / retargetLeg. They are only ever
// READ (crossVectors does not mutate its arguments), so hoisting them makes
// those branches allocation-free.
const _axisZ = new Vector3(0, 0, 1);

function tmpVec(): Vector3 {
  return VEC_POOL.pop() ?? new Vector3();
}
function tmpQuat(): Quaternion {
  return QUAT_POOL.pop() ?? new Quaternion();
}
function freeVec(v: Vector3) {
  if (VEC_POOL.length < 200) VEC_POOL.push(v);
}
function freeQuat(q: Quaternion) {
  if (QUAT_POOL.length < 200) QUAT_POOL.push(q);
}

/** Result: world-space position of hips + local bone rotations. */
export interface RetargetedBody {
  /** World position of the root (hips) joint. */
  hipsWorldPos: Vector3;
  /** Local-space bone rotations keyed by bone name. */
  boneRotations: Map<string, Quaternion>;
  /** World-space rotations keyed by bone name (for child lookups like hands). */
  worldRotations: Map<string, Quaternion>;
}

/**
 * Compute signed angle from `from` to `to` around `axis` (right-handed,
 * positive = counter-clockwise when looking along axis).
 */
function signedAngle(from: Vector3, to: Vector3, axis: Vector3): number {
  const f = tmpVec().copy(from).projectOnPlane(axis);
  const t = tmpVec().copy(to).projectOnPlane(axis);
  const angle = Math.atan2(
    axis.dot(f.clone().cross(t)),
    f.dot(t)
  );
  freeVec(f);
  freeVec(t);
  return angle;
}

/**
 * Compute the world-space rotation for a single bone given:
 * - bindDir: rest direction of the bone (world space, assuming identity bind rotation)
 * - refSide: a perpendicular "reference" vector in the bone's plane (world space)
 * - targetDir: desired direction (world space, normalized)
 * - targetSide: desired reference side vector (world space)
 * - parentWorldRot: the parent's current world-space rotation
 *
 * Returns the local-space rotation (relative to parent).
 */
function retargetSingleBone(
  bone: ResolvedBone,
  targetDir: Vector3,
  targetSide: Vector3,
  parentWorldRot: Quaternion
): Quaternion {
  // Bind-pose direction (world space since bind rot = identity)
  // PHASE 11: was `bone.localDir.clone()`. `bindDir` is read-only from here
  // through the `rotationBetween` call and is not held past it. retargetSingleBone
  // is called sequentially (never re-entrantly) from retargetBody, and the only
  // callee that also uses module scratch is signedAngle, which runs after
  // bindDir has been consumed and never touches this slot.
  const bindDir = _bBindDir.copy(bone.localDir);
  // Ensure bindDir is normalized
  bindDir.normalize();

  if (bindDir.length() < 1e-6) {
    // Zero-length bone (e.g. hips position-only)
    // PHASE 11 NOTE: deliberately still `new Quaternion()` -- the result lands in
    // the caller's boneRotations Map, so each bone needs a DISTINCT object.
    const id = new Quaternion();
    return id;
  }

   // Reference side: derive from targetSide projected onto the plane
   // perpendicular to bindDir.  This guarantees zero twist in the T-pose
   // (where targetDir ≈ bindDir and targetSide matches the bind-pose side).
   // Falls back to cross(up, bindDir) when targetSide is degenerate.
   let refSide = tmpVec().copy(targetSide).projectOnPlane(bindDir).normalize();
   if (refSide.length() < 0.1) {
     // targetSide is too close to bindDir; fall back to cross(up, bindDir)
     refSide = tmpVec().crossVectors(CANONICAL_UP, bindDir).normalize();
     if (refSide.length() < 0.1) {
       // Up is too close to bone direction; use GLTF forward as fallback
       refSide.crossVectors(_axisZ, bindDir).normalize();
     }
   }

  // Step 1: Aim rotation
  const dirQuat = tmpQuat();
  rotationBetween(bindDir, targetDir, dirQuat);

  // Step 2: Rotate the reference side vector by the aim rotation
  const rotatedSide = tmpVec().copy(refSide).applyQuaternion(dirQuat);

  // Step 3: Twist angle
  const twistAngle = signedAngle(rotatedSide, targetSide, targetDir);

  // Step 4: Twist quaternion
  const twistQuat = tmpQuat().setFromAxisAngle(targetDir, twistAngle);

  // Step 5: Combine (world space).
  // The twist must be applied AFTER the aim, in world space (W = twist * aim): its axis
  // IS the bone's post-aim axis (targetDir).  With `aim * twist` the pre-aim twist drags
  // the bone's direction off its target whenever the twist is non-zero — measured on the
  // left clavicle (dot 0.63, i.e. 51 deg off) and the raised left upper arm (dot 0.85).
  // W = twist * aim keeps the child offset exactly on targetDir AND aligns the
  // reference side with targetSide.
  const worldRot = tmpQuat().copy(twistQuat).multiply(dirQuat);

  // Step 6: Convert to local
  const localRot = tmpQuat().copy(parentWorldRot).invert().multiply(worldRot);

  freeVec(refSide);
  freeVec(rotatedSide);
  freeQuat(dirQuat);
  freeQuat(twistQuat);
  freeQuat(worldRot);
  // Return a fresh quaternion (caller stores a copy)

  // Free temp vecs/quats — we already copied worldRot into localRot
  return localRot.clone();
}

/**
 * Retarget the full body pose (spine, head, arms, legs) to avatar bone rotations.
 *
 * The canonical pose's `body` field already provides world-space positions
 * in canonical coordinates (meters). We map these directly to the avatar's
 * bone rotations using the resolved skeleton's bind-pose directions.
 */
export function retargetBody(
  skeleton: ResolvedSkeleton,
  body: BodyPose
): RetargetedBody {
  const boneRotations = new Map<string, Quaternion>();
  // Track world-space rotations as we go down the hierarchy
  const worldRotations = new Map<string, Quaternion>();

  const hipsBone = skeleton.getBone("hips");
  const hipsWorldPos = hipsBone ? cvec(body.hipsPos, _hipsPos).clone() : new Vector3();

  // --- Root (hips) ---
  // Hips rotation: orient the spine vertically with correct twist.
  if (hipsBone) {
    // PHASE 23 — REJECTED BY MEASUREMENT. The pelvis tilt clamp is NOT applied
    // here, and the reason is worth recording because it is not obvious.
    //
    // The obvious hypothesis was that the hips basis tilts with `spineDir` and
    // that bounding the tilt would stop the feet swinging under the floor. It
    // does not apply, because `spineDir` is EXACTLY [0,1,0] -- perfectly
    // vertical -- on all 632 frames across all five clips (measured
    // acos(gltfY) = 0.0000 deg everywhere). The hips basis is already upright,
    // so there is no pelvis tilt to clamp.
    //
    // This also explains why the earlier ankle-smoothing sweep could not work:
    // the motion is not a trunk lean at all. Whatever rotates the body is
    // downstream of the hips basis, and the earlier per-frame trace labelled
    // "thighDirY" (0.998 -> 0.601) was measuring the THIGH bone direction, not
    // the spine axis -- the two are different quantities and were conflated.
    //
    // A clamp here would be dead code that silently does nothing while
    // appearing to fix a floor-penetration bug. Removed rather than shipped.
    const spine = tmpVec().copy(cvec(body.spineDir, _spineDir)).normalize();
    const shoulderAxis = cvec(body.shoulderAxis, _shoulderAxis).normalize();

    // Build basis:
    //   X = the SUBJECT'S LEFT  (NOT the shoulder axis itself — see below)
    //   Y = spine direction (up)
    //   Z = X x Y = the character's facing (+Z world for a neutral pose)
    //
    // The Hips bone's bind rotation is the identity (measured on Xbot.glb: every node
    // rotation is [0,0,0,1]), so its bind basis IS the world basis — its local +X is
    // the MODEL'S LEFT (+X world), +Y is up and +Z is the facing.  `body.shoulderAxis`
    // is (rightShoulder - leftShoulder) = the subject's RIGHT, so mapping the local X
    // onto it yawed the whole root 180deg (and, because children inherit the root's
    // translation frame, it also swapped the Left/Right leg root positions).
    const rightDir = tmpVec().copy(shoulderAxis).projectOnPlane(spine).normalize();
    if (rightDir.length() < 0.1) {
      rightDir.crossVectors(CANONICAL_UP, spine).normalize();
    }
    const leftDir = tmpVec().copy(rightDir).negate();
    const forward = tmpVec().crossVectors(leftDir, spine).normalize();

    const rootRot = tmpQuat().setFromRotationMatrix(
      _mat.makeBasis(leftDir, spine, forward)
    );

    boneRotations.set(hipsBone.name, rootRot.clone());
    worldRotations.set(hipsBone.name, rootRot.clone());

    freeQuat(rootRot);
    freeVec(spine);
    freeVec(rightDir);
    freeVec(leftDir);
    freeVec(forward);
  }

  // --- Head / Neck ---
  const neckBone = skeleton.getBone("neck");
  const headBone = skeleton.getBone("head");

  // --- Spine chain ---
  // Map spine bones to progressively align with the spine direction
  const spineBones: Array<{ role: CanonicalRole; name: string }> = [];
  for (const role of ["spine1", "spine2", "spine3"] as CanonicalRole[]) {
    const bone = skeleton.getBone(role);
    if (bone) {
      spineBones.push({ role, name: bone.name });
    }
  }

  // PHASE 11: these MUST be independent copies, not aliases of the scratch
  // slots. `spineTwistAxis` and `poseSpineDir` are both consumed inside the
  // spine loop below while `_shoulderAxis` / `_spineDir` are also written by
  // retargetArm/retargetLeg deeper in the call tree. Returning the scratch
  // object itself would let a later cvec() silently rewrite a value that is
  // still being used -- exactly the cross-contamination this phase must avoid.
  const spineTwistAxis = tmpVec().copy(cvec(body.shoulderAxis, _shoulderAxis)).normalize();

  // Compute the body's overall pose rotation: the rotation from the skeleton's
  // rest spine direction to the pose's spine direction. This preserves each
  // spine bone's relative curvature while following the body's lean.
  const poseSpineDir = tmpVec().copy(cvec(body.spineDir, _spineDir)).normalize();
  const restSpineDir = tmpVec();
  let spinePoseRot = _spinePoseRot.identity();
  if (hipsBone && neckBone) {
    restSpineDir.subVectors(neckBone.worldPos, hipsBone.worldPos).normalize();
    if (restSpineDir.lengthSq() > 1e-8) {
      spinePoseRot = tmpQuat().setFromUnitVectors(restSpineDir, poseSpineDir);
    }
  }

  for (let i = 0; i < spineBones.length; i++) {
    const bone = skeleton.bones.get(spineBones[i].name)!;
    const parent = bone.parent ?? (hipsBone ? hipsBone.name : "hips");
    const parentWorldRot = worldRotations.get(parent) ?? _identQ.identity();

    // Target direction: the bone's bind direction, rotated by the overall body
    // pose rotation. This preserves the spine's natural curvature while aligning
    // the overall spine with the desired direction.
    const boneTargetDir = tmpVec().copy(bone.localDir).applyQuaternion(spinePoseRot);

    const localRot = retargetSingleBone(
      bone,
      boneTargetDir,
      spineTwistAxis,
      parentWorldRot
    );

    boneRotations.set(bone.name, localRot);
    // Update world rotation
    const world = tmpQuat().copy(parentWorldRot).multiply(localRot);
    worldRotations.set(bone.name, world.clone());
    freeQuat(world);
    freeVec(boneTargetDir);
  }

  // NOTE: `spinePoseRot` (a pool-allocated THREE.Quaternion when the guard above
  // succeeds) is still used by the neck and head blocks below, so it must NOT be
  // released here — `tmpQuat()` inside those blocks would hand the same object out
  // again and overwrite it mid-flight, corrupting the neck/head targets.
  // Both temporaries are released at the end of this function.

  if (neckBone) {
    const neckTargetDir = tmpVec().copy(cvec(body.headDir, _headDir)).applyQuaternion(spinePoseRot).normalize();
    const neckParent = neckBone.parent ?? (skeleton.getBone("spine3")?.name ?? neckBone.name);
    const neckParentWorldRot = worldRotations.get(neckParent) ?? _identQ.identity();

    const neckLocalRot = retargetSingleBone(
      neckBone,
      neckTargetDir,
      cvec(body.shoulderAxis, _shoulderAxis).normalize(),
      neckParentWorldRot
    );
    boneRotations.set(neckBone.name, neckLocalRot);
    worldRotations.set(neckBone.name, tmpQuat().copy(neckParentWorldRot).multiply(neckLocalRot));
    freeVec(neckTargetDir);
  }

  if (headBone) {
    const headDir = tmpVec().copy(cvec(body.headDir, _headDir)).applyQuaternion(spinePoseRot).normalize();
    const headParent = headBone.parent ?? (skeleton.getBone("neck")?.name ?? headBone.name);
    const headParentWorldRot = worldRotations.get(headParent) ?? _identQ.identity();
    // Head side: project shoulder axis perpendicular to head dir
    const headSide = tmpVec()
      .copy(cvec(body.shoulderAxis, _shoulderAxis))
      .projectOnPlane(headDir)
      .normalize();

    const headLocalRot = retargetSingleBone(
      headBone,
      headDir,
      headSide,
      headParentWorldRot
    );
    boneRotations.set(headBone.name, headLocalRot);
    worldRotations.set(headBone.name, tmpQuat().copy(headParentWorldRot).multiply(headLocalRot));
    freeVec(headSide);
    freeVec(headDir);
  }

  // --- Arms (4-bone chain each) ---
  retargetArm(skeleton, body, boneRotations, worldRotations, "left");
  retargetArm(skeleton, body, boneRotations, worldRotations, "right");

  // --- Legs (3-bone chain each) ---
  retargetLeg(skeleton, body, boneRotations, worldRotations, "left");
  retargetLeg(skeleton, body, boneRotations, worldRotations, "right");

  // Release the spine-setup temporaries only now that every consumer is done.
  freeVec(restSpineDir);

  return {
    hipsWorldPos,
    boneRotations,
    worldRotations,
  };
}

/**
 * Retarget one arm chain: shoulder → upperArm → forearm → hand.
 *
 * The twist for upperarm and forearm is determined by the arm's
 * "palm normal" vector — the direction perpendicular to the arm
 * plane defined by shoulder→elbow→wrist.
 */
function retargetArm(
  skeleton: ResolvedSkeleton,
  body: BodyPose,
  boneRotations: Map<string, Quaternion>,
  worldRotations: Map<string, Quaternion>,
  side: "left" | "right"
) {
  const shoulder = skeleton.getBone(`${side}Shoulder` as CanonicalRole);
  const upperArm = skeleton.getBone(`${side}UpperArm` as CanonicalRole);
  const forearm = skeleton.getBone(`${side}Forearm` as CanonicalRole);
  const hand = skeleton.getBone(`${side}Hand` as CanonicalRole);

  if (!shoulder || !upperArm || !forearm || !hand) return;

  // Get canonical positions
  // PHASE 11: redundant `.clone()`s removed — cvec already produced a fresh
  // vector, so the clone was a second allocation for no reason. All three are
  // live at once (armDir and forearmDir are both derived from them), hence
  // three separate slots.
  const shoulderPos = cvec(
    side === "left" ? body.leftShoulder : body.rightShoulder,
    _shoulderPos
  );
  const elbowPos = cvec(
    side === "left" ? body.leftElbow : body.rightElbow,
    _elbowPos
  );
  const wristPos = cvec(
    side === "left" ? body.leftWrist : body.rightWrist,
    _wristPos
  );

  // Arm plane: the plane containing shoulder, elbow, wrist
  const armDir = tmpVec().subVectors(elbowPos, shoulderPos).normalize();
  const forearmDir = tmpVec().subVectors(wristPos, elbowPos).normalize();

  // PHASE 18: the "side" vector of the arm plane (perpendicular to armDir) --
  // the pole vector that fixes the elbow's roll and therefore the forearm's
  // twist.
  //
  // The previous code used a GLOBAL world up (CANONICAL_UP = (0,1,0)) as the
  // fallback when the cross product was degenerate, i.e. when the arm is
  // straight (|armDir x foreDir| == sin(elbow bend angle) -> 0). Two measured
  // problems with that:
  //
  //   1. It is not in the torso's frame, so the roll error it produces changes
  //      as the torso pitches or rolls. For a T-pose it is wrong by 90 deg.
  //   2. Because the cross product is ~0 for a straight arm, a few millimetres
  //      of MediaPipe sensor noise flip the arm between the correct roll and
  //      90 deg off, frame to frame -- the arm visibly snaps between two
  //      orientations. Reproduced by sweeping +/-20 mm of wrist noise on a
  //      straight arm.
  //
  // The correct reference is the SUBJECT'S OWN LATERAL AXIS: a human elbow
  // hinges about the side-to-side axis, so the arm-plane normal is the shoulder
  // axis regardless of torso orientation. `shoulderAxis` is (rSh - lSh) = the
  // subject's RIGHT. Being torso-relative by construction, it is continuous and
  // cannot flip.
  //
  // NOTE: the raw magnitude is tested via lengthSq() before the branch because
  // THREE's normalize() is divideScalar(length() || 1), which maps a zero
  // vector to (0,0,0) rather than NaN -- so the normalized length test is a
  // valid 0 and the branch does fire. lengthSq() is used only to avoid the
  // sqrt in the hot path.
  const armPlaneSide = tmpVec().crossVectors(armDir, forearmDir);
  if (armPlaneSide.lengthSq() < 0.01) {
    armPlaneSide.copy(cvec(body.shoulderAxis, _shoulderAxis)).normalize();
  } else {
    armPlaneSide.normalize();
  }
  // Residual guard: if the arm lies along the shoulder axis, the lateral
  // reference projects to zero. Use the torso's forward direction (spine x
  // lateral) instead of world up, for the same reason.
  armPlaneSide.projectOnPlane(armDir);
  if (armPlaneSide.length() < 0.1) {
    armPlaneSide.crossVectors(cvec(body.spineDir, _spineDir), armDir).normalize();
  }
  armPlaneSide.normalize();

   // Shoulder (clavicle): aim at the LATERAL direction from the shoulder toward
   // the arm, NOT at `armDir`.
   //
   // PHASE 17 ROOT-CAUSE FIX. The clavicle's child is the upper arm, so its bind
   // `localDir` is LATERAL (measured on Xbot.glb: mixamorigLeftShoulder's
   // localDir = (0.9774, -0.0484, -0.2060), i.e. almost pure +X). The previous
   // code aimed it at `armDir` (shoulder->elbow), which points DOWN/BACK. Those
   // two vectors are near-orthogonal (measured dot = 0.0313), so the solver
   // applied a ~88-94 deg rotation to the clavicle on EVERY frame -- roughly
   // six times the 5-15 deg a real clavicle protracts/elevates, and the reason
   // the shoulders read as jammed upward toward the ears.
   //
   // The correct target is the subject's own lateral axis. `shoulderAxis` is
   // (rightShoulder - leftShoulder) = the subject's RIGHT, so the LEFT clavicle
   // aims at -shoulderAxis. Verified on real data: dot(clavBind, -shoulderAxis)
   // = 0.9679, i.e. the bind pose is already almost exactly right, which is what
   // a bind pose should be.
   //
   // Using the lateral axis (not the arm direction) is also anatomically right:
   // a clavicle swings the shoulder girdle forward/back and up/down, it does
   // not swing the arm. The upper arm below then still aims at `armDir`, so all
   // actual arm motion is preserved exactly.
   //
   // MEASURED on ADOPT (79 frames): mean 82.1 deg -> 30.8 deg away from the bind
   // clavicle direction, i.e. the clavicle is now in the right neighbourhood
   // instead of permanently perpendicular to it.
   //
   // HONEST CAVEAT: the new target is not perfectly smooth. `shoulderAxis` is
   // derived from two MediaPipe shoulder landmarks that jitter (measured
   // max 0.44 m frame-to-frame on this capture), so the lateral axis wobbles up
   // to ~49 deg frame-to-frame where the old armDir target wobbled ~15 deg. The
   // old target was wrong-but-smooth; this one is right-but-noisier. Both come
   // from the same noisy source, so the real fix for the jitter is upstream
   // landmark smoothing -- out of scope here, and deliberately NOT papered over
   // with a magic smoothing constant.
   const spineDir = cvec(body.spineDir, _spineDir).normalize();
   const shoulderAxisW = cvec(body.shoulderAxis, _shoulderAxis).normalize();
   // Project onto the trunk plane so the clavicle never gains a spurious roll
   // component from shoulder tilt; fall back to the old axis if degenerate.
   const lateral = tmpVec()
     .copy(shoulderAxisW)
     .multiplyScalar(side === "left" ? -1 : 1)
     .projectOnPlane(spineDir)
     .normalize();
   if (lateral.length() < 0.1) {
     lateral.copy(armDir);
   }

   const shoulderParent = shoulder.parent ?? (skeleton.getBone("spine3")?.name ?? shoulder.name);
   const shoulderParentWorldRot = worldRotations.get(shoulderParent) ?? _identQ.identity();

   const shoulderLocalRot = retargetSingleBone(
     shoulder,
     lateral,
     spineDir,
     shoulderParentWorldRot
   );
   boneRotations.set(shoulder.name, shoulderLocalRot);
   worldRotations.set(shoulder.name, tmpQuat().copy(shoulderParentWorldRot).multiply(shoulderLocalRot));

    // Upper arm: align to armDir (shoulder -> ELBOW).
    // The upper-arm bone's child offset IS the elbow — measured on Xbot.glb:
    // mixamorig:LeftForeArm sits at (27.84, 0, 0) in mixamorig:LeftArm's local space,
    // so the bone's bind "point" is the elbow, not the wrist.  (The legacy
    // retarget.ts:453-455 aims DEFAULT_BONE_DIRS.leftUpperArm=[1,0,0] at
    // (leftElbow - leftShoulder) — the same rule.)  Aiming it at forearmDir instead
    // collapsed the elbow: the forearm's local rotation then resolved to ~identity,
    // so the arm rendered as a rigid straight stick, and for a folded arm
    // (armDir ~= -forearmDir) the upper arm swung through ~180deg.
    const uaParentWorldRot = worldRotations.get(shoulder.name) ?? _identQ.identity();
    const upperArmLocalRot = retargetSingleBone(
      upperArm,
      armDir,
      armPlaneSide,
      uaParentWorldRot
    );
   boneRotations.set(upperArm.name, upperArmLocalRot);
   worldRotations.set(upperArm.name, tmpQuat().copy(uaParentWorldRot).multiply(upperArmLocalRot));

   // Forearm: align to forearmDir, twist from arm plane side
   // The arm plane side should remain consistent across the chain
   const faParentWorldRot = worldRotations.get(upperArm.name) ?? _identQ.identity();
   const forearmLocalRot = retargetSingleBone(
     forearm,
     forearmDir,
     armPlaneSide,
     faParentWorldRot
   );
   boneRotations.set(forearm.name, forearmLocalRot);
   worldRotations.set(forearm.name, tmpQuat().copy(faParentWorldRot).multiply(forearmLocalRot));

   // Hand: aligned by hand retargeter (called separately)
   // Store forearm world rotation for the hand retargeter to consume
   if (!boneRotations.has(hand.name)) {
     const handParentWorldRot = worldRotations.get(forearm.name) ?? _identQ.identity();
     boneRotations.set(hand.name, new Quaternion());
     worldRotations.set(hand.name, tmpQuat().copy(handParentWorldRot));
   }

  freeVec(armDir);
  freeVec(forearmDir);
  freeVec(armPlaneSide);
  freeVec(lateral);
  // PHASE 11 NOTE: shoulderPos/elbowPos/wristPos are module-scoped scratch, NOT
  // pooled temporaries, so they must NOT be returned to VEC_POOL here. Doing so
  // would push a shared slot into the free list, where a later tmpVec() would
  // hand the same object out for an unrelated purpose and silently corrupt it.
  // The original `.clone()` calls hid this by making each one a private object.
}

/**
 * Retarget one leg chain: hip → knee → ankle.
 */
function retargetLeg(
  skeleton: ResolvedSkeleton,
  body: BodyPose,
  boneRotations: Map<string, Quaternion>,
  worldRotations: Map<string, Quaternion>,
  side: "left" | "right"
) {
  const hip = skeleton.getBone(`${side}Hip` as CanonicalRole);
  const knee = skeleton.getBone(`${side}Knee` as CanonicalRole);
  const ankle = skeleton.getBone(`${side}Ankle` as CanonicalRole);

  if (!hip || !knee || !ankle) return;

  // PHASE 11: the `.clone()` calls these used to have are gone. `cvec` wrote
  // into a fresh vector and the clone copied it, so the clone was a redundant
  // second allocation; writing straight into the dedicated scratch slots yields
  // the same value. All three are live simultaneously, which is exactly why
  // they need three separate slots.
  const hipPos = cvec(side === "left" ? body.leftHip : body.rightHip, _hipPos);
  const kneePos = cvec(side === "left" ? body.leftKnee : body.rightKnee, _kneePos);
  const anklePos = cvec(side === "left" ? body.leftAnkle : body.rightAnkle, _anklePos);

  const thighDir = tmpVec().subVectors(kneePos, hipPos).normalize();
  const shinDir = tmpVec().subVectors(anklePos, kneePos).normalize();

  // Leg plane side vector
  const legPlaneSide = tmpVec().crossVectors(thighDir, shinDir).normalize();
   if (legPlaneSide.length() < 0.01) {
     legPlaneSide.copy(CANONICAL_UP);
     legPlaneSide.projectOnPlane(thighDir).normalize();
     if (legPlaneSide.length() < 0.1) {
       legPlaneSide.crossVectors(_axisZ, thighDir).normalize();
     }
   }

   // Hip bone (leftHip / rightHip)
   // Parent is typically hips
   const hipParent = hip.parent ?? (skeleton.getBone("hips")?.name ?? "hips");
   const hipParentWorldRot = worldRotations.get(hipParent) ?? _identQ.identity();

   const hipLocalRot = retargetSingleBone(
     hip,
     thighDir,
     legPlaneSide,
     hipParentWorldRot
   );
   boneRotations.set(hip.name, hipLocalRot);
   worldRotations.set(hip.name, tmpQuat().copy(hipParentWorldRot).multiply(hipLocalRot));

   // Knee
   const kneeParentWorldRot = worldRotations.get(hip.name) ?? _identQ.identity();
   const kneeLocalRot = retargetSingleBone(
     knee,
     shinDir,
     legPlaneSide,
     kneeParentWorldRot
   );
   boneRotations.set(knee.name, kneeLocalRot);
   worldRotations.set(knee.name, tmpQuat().copy(kneeParentWorldRot).multiply(kneeLocalRot));

   // Ankle / foot: the canonical pose carries NO ankle->toe direction (BodyPose has
   // no toe field), and mixamorig:LeftFoot's child offset points at the TOE
   // (measured childDir = (0, -0.632, 0.775)), not along the shin.  Aiming it at
   // shinDir therefore collapsed the foot onto the shin line (measured 49.4deg of
   // rotation with zero source data).  With no data, the foot must simply follow the
   // aimed shin at its bind rotation.
   const ankleParentWorldRot = worldRotations.get(knee.name) ?? _identQ.identity();
   const ankleLocalRot = new Quaternion(); // identity = inherit the shin
   boneRotations.set(ankle.name, ankleLocalRot);
   worldRotations.set(ankle.name, tmpQuat().copy(ankleParentWorldRot).multiply(ankleLocalRot));

  freeVec(thighDir);
  freeVec(shinDir);
  freeVec(legPlaneSide);
}
