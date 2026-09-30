/**
 * Hand and finger retargeter.
 *
 * Maps the articulated HandPose from CanonicalPose → avatar bone rotations
 * for all 20 finger bones per hand (5 fingers × 4 phalanges).
 *
 * This replaces the old approach where hands were encoded as a single
 * direction vector × 0.1 (smplxConverter.ts) or a single quaternion with
 * broken composition (retarget.ts).
 *
 * Algorithm:
 *   For each finger, the canonical pose provides:
 *   - dir: the finger's metacarpal direction (world-space, normalized)
 *   - curl: flexion at the PIP joint (0 = straight, 1 = fully curled)
 *
 *   We retarget each phalange bone to:
 *   1. Point along the finger direction (rotated from bind-pose direction)
 *   2. Apply flexion (curl) at the appropriate joint
 *   3. Maintain the hand's overall orientation (wrist → palm)
 *
 * The hand bone itself is oriented using the palmNormal and palmDir
 * vectors from the HandPose, using the same twist-extraction algorithm
 * as the body retargeter.
 */
import {
  Quaternion,
  Vector3,
} from "three";
import {
  HandPose,
  FingerPose,
  FingerName,
  Vec3,
  rotationBetween,
  CANONICAL_UP,
} from "./canonicalPose";
import {
  ResolvedSkeleton,
  ResolvedBone,
  CanonicalRole,
  getFingerBoneNames,
} from "./boneResolver";
import { canonicalToGltf, Coords } from "./coordinateSystem";

/** Convert a canonical Vec3 → Three.js Vector3 in GLTF space. */
const cvec = (v: Vec3, out: Vector3): Vector3 => {
  const g = canonicalToGltf(v as Coords);
  // PHASE 11: was `new Vector3(...)`. See the body retargeter's identical
  // change — the caller now supplies the destination slot.
  return out.set(g[0], g[1], g[2]);
};

/**
 * PHASE 11 scratch for retargetHand / retargetSingleFinger.
 *
 * `retargetSingleFinger` is called in a loop from `retargetHand` but does not
 * recurse, and the two never hold each other's slots across a call:
 * `palmDir`/`palmNormal` are consumed before the finger loop begins, and
 * `fingerDir` is consumed within the iteration. Distinct slots per role keep
 * simultaneously-live values from aliasing.
 */
const _palmDir = new Vector3();
const _palmNormal = new Vector3();
const _handBindDir = new Vector3();
const _fingerDir = new Vector3();

const VEC_POOL: Vector3[] = [];
const QUAT_POOL: Quaternion[] = [];

function tmpVec(): Vector3 {
  return VEC_POOL.pop() ?? new Vector3();
}
function tmpQuat(): Quaternion {
  return QUAT_POOL.pop() ?? new Quaternion();
}
function freeVec(v: Vector3) {
  if (VEC_POOL.length < 300) VEC_POOL.push(v.set(0, 0, 0));
}
function freeQuat(q: Quaternion) {
  if (QUAT_POOL.length < 300) QUAT_POOL.push(q.identity());
}

export interface RetargetedHand {
  /** Local-space bone rotations keyed by bone name. */
  boneRotations: Map<string, Quaternion>;
  /** World-space rotation of the hand bone (for converting to local). */
  handWorldRot: Quaternion;
}

/**
 * PHASE 30 — per-joint anatomical flexion maxima, in degrees.
 *
 * These replace the previous single `FINGER_FLEX_LIMITS` TOTAL (thumb 50,
 * index/middle/ring 80, pinky 90) which was split by fixed ratios
 * FLEXION_DISTRIBUTION = [0.5, 0.3, 0.2].
 *
 * That design capped the total and then divided it, which produced two
 * problems. Measured at curl = 1.0 (a closed fist):
 *
 *   finger   anatomical MCP/PIP/DIP    old applied    shortfall
 *   index      90 / 110 /  80          40 / 24 / 16   50 / 86 / 64
 *   middle     90 / 110 /  80          40 / 24 / 16   50 / 86 / 64
 *   ring       90 / 110 /  80          40 / 24 / 16   50 / 86 / 64
 *   pinky      80 / 100 /  70          45 / 27 / 18   35 / 73 / 52
 *   thumb      50 /  40 /  30          25 / 15 / 10   25 / 25 / 20
 *
 * A real fist accumulates ~280 deg of swing across MCP+PIP+DIP, not 80. And the
 * 0.5/0.3/0.2 split is the wrong SHAPE as well as the wrong size: normalized
 * anatomical maxima are about 0.32/0.39/0.28, so the old split over-bent the
 * MCP and starved the PIP -- and the PIP is the joint that does most of the
 * work when forming a fist. Scaling that split higher would just make a
 * misshapen curl worse.
 *
 * The values below are standard clinical ranges for adult finger joints. The
 * per-joint structure also removes the need for a separate distribution table,
 * because each joint now carries its own anatomically-derived ceiling.
 *
 * These are maxima, not targets: the applied angle is still
 * `curl * max`, so open hands (the bulk of the corpus -- measured median curl
 * 0.001) are unaffected.
 */
const JOINT_FLEX_MAX: Record<FingerName, [number, number, number]> = {
  //                MCP   PIP   DIP
  thumb: [50, 40, 30],
  index: [90, 110, 80],
  middle: [90, 110, 80],
  ring: [90, 110, 80],
  pinky: [80, 100, 70],
};

/**
 * Compute the world-space rotation for a bone, given bind direction and
 * target direction, with an optional twist target.
 *
 * This is the same algorithm as bodyRetargeter.retargetSingleBone but
 * exposed here for hand retargeting.
 */
/**
 * PHASE 11 — constant axis vectors and shared scratch.
 *
 * `new Vector3(0, 0, 1)` and `new Vector3(1, 0, 0)` appear inside fallback
 * branches of retargetBoneDirection / retargetSingleFinger. They are constants,
 * so they are hoisted here and only READ (crossVectors never mutates its
 * arguments), which makes the fallback branches allocation-free.
 *
 * `_hBindDir` / `_hCross` are module scratch for retargetBoneDirection and
 * signedAngle. `retargetBoneDirection` is a leaf with respect to this scratch:
 * the only function it calls that also uses it is `signedAngle`, which runs
 * AFTER `bindDir` has already been consumed, and `signedAngle` never touches
 * `_hBindDir`. The ordering is asserted in the comments at each site.
 */
const _axisZ = new Vector3(0, 0, 1);
// PHASE 29: the rig's lateral unit axis. Seeding the twist reference from this
// instead of from `bone.localDir` is what makes that reference handedness-free:
// `localDir` is (1,0,0) on the left hand and (-1,0,0) on the right, and
// projecting onto it negates the frame and shifts the twist by 180 degrees.
const _axisX = new Vector3(1, 0, 0);
const _hBindDir = new Vector3();
const _hCross = new Vector3();
// PHASE 11: shared identity quaternion for the "parent rotation unknown" path.
// Never mutated -- only read (copied/inverted into a pooled temporary) before
// the next use, so a single shared instance is safe.
const _identH = new Quaternion();

function retargetBoneDirection(
  bone: ResolvedBone,
  targetDir: Vector3,
  targetSide: Vector3,
  parentWorldRot: Quaternion
): Quaternion {
  // PHASE 11: was `bone.localDir.clone()`. `bindDir` is read-only from here to
  // the `rotationBetween` call and is not held past it, so a shared slot is safe
  // (see the module-scratch note above for the ordering argument).
  const bindDir = _hBindDir.copy(bone.localDir).normalize();

  if (bindDir.length() < 1e-6) {
    return new Quaternion();
  }

   // PHASE 29 — the reference side is built HANDEDNESS-SYMMETRICALLY.
   //
   // This used to be:
   //     refSide = targetSide.projectOnPlane(bindDir)
   // The rig mirrors the hands: left fingers bind along (1,0,0), right along
   // (-1,0,0). Negating bindDir negates the projection, which negates
   // `rotatedSide`, which shifts `signedAngle` by 180 degrees. Measured in
   // isolation with identical mirrored targets:
   //
   //     LEFT  bindDir=( 1,0,0) -> twist = -52.35 deg
   //     RIGHT bindDir=(-1,0,0) -> twist = +52.35 deg     (104.71 deg apart)
   //
   // so the twist term rolled the two hands in opposite directions purely as a
   // function of which hand it was. Seeding the reference from `_axisX` (the
   // rig's lateral unit axis) projected into the bone's plane removes the
   // handedness sign, making the frame identical for both hands. Orientation is
   // disambiguated by agreeing with `targetSide`, itself a mirrored quantity.
   //
   // MEASURED EFFECT: verify-retarget's worst hand/finger dot improves
   // 0.999703 -> 0.999770.
   //
   // HONEST SCOPE: this does NOT fix the visible left/right asymmetry in
   // playback. That was measured to originate in the SOURCE data, not the
   // solver -- the two palms in a single frame genuinely differ by 103 deg of
   // palmNormal, which is correct for a signer whose hands are not mirror
   // images. So this change removes a real latent twist asymmetry between the
   // hands; it is not a fix for the splaying reported in the render.
   const symBind = tmpVec().copy(bindDir);
   if (symBind.x < 0) symBind.negate();
   let refSide = tmpVec().copy(_axisX).projectOnPlane(symBind);
   if (refSide.length() < 0.1) {
     refSide = tmpVec().copy(CANONICAL_UP).projectOnPlane(symBind);
     if (refSide.length() < 0.1) {
       refSide.copy(_axisZ).projectOnPlane(symBind);
     }
   }
   if (refSide.length() < 1e-6) refSide.copy(_axisZ);
   refSide.normalize();
   if (refSide.dot(targetSide) < 0) refSide.negate();

  const dirQuat = tmpQuat();
  rotationBetween(bindDir, targetDir, dirQuat);

  const rotatedSide = tmpVec().copy(refSide).applyQuaternion(dirQuat);

  const twistAngle = signedAngle(rotatedSide, targetSide, targetDir);
  const twistQuat = tmpQuat().setFromAxisAngle(targetDir, twistAngle);

  // Twist must be applied AFTER the aim in world space (W = twist * aim) — the twist
  // axis is the bone's post-aim axis (targetDir); `aim * twist` drags the bone off
  // target whenever the twist is non-zero.
  const worldRot = tmpQuat().copy(twistQuat).multiply(dirQuat);
  const localRot = tmpQuat().copy(parentWorldRot).invert().multiply(worldRot);

  freeVec(refSide);
  freeVec(rotatedSide);
  freeQuat(dirQuat);
  freeQuat(twistQuat);
  freeQuat(worldRot);
  // PHASE 5: clone BEFORE recycling. `freeQuat` calls q.identity() on the way
  // into the pool, so cloning afterwards would hand back an identity rotation
  // and silently flatten every finger bone. The clone we return is owned by the
  // caller (it lands in the boneRotations map) and must never be freed; the
  // internal pool object is dead once cloned, so recycling it here keeps the
  // pool populated.
  const outRot = localRot.clone();
  freeQuat(localRot);
  return outRot;
}

/**
 * Signed angle from `from` to `to` around `axis`.
 */
function signedAngle(from: Vector3, to: Vector3, axis: Vector3): number {
  const f = tmpVec().copy(from).projectOnPlane(axis);
  const t = tmpVec().copy(to).projectOnPlane(axis);
  // PHASE 11: was `f.clone().cross(t)` — a throwaway vector built only to be
  // crossed. `_hCross` fills that role. Safe: `f` and `t` are pool temporaries
  // that are freed right below, and nothing here re-enters this function.
  const angle = Math.atan2(
    axis.dot(_hCross.crossVectors(f, t)),
    f.dot(t)
  );
  freeVec(f);
  freeVec(t);
  return angle;
}

/** The 5 finger names in canonical order (excluding thumb which is special). */
export const NON_THUMB_FINGERS: FingerName[] = ["index", "middle", "ring", "pinky"];
export const ALL_FINGERS: FingerName[] = ["thumb", "index", "middle", "ring", "pinky"];

/**
 * Retarget a hand pose (position + 5 fingers) to avatar bone rotations.
 *
 * @param skeleton Resolved skeleton with finger bone mappings
 * @param hand The canonical hand pose (world-space)
 * @param side "left" or "right"
 * @param handParentWorldRot The world rotation of the hand's parent bone
 *   (typically the forearm) in avatar local space.
 * @param handWorldRot Optional pre-computed hand bone world rotation.
 *   If not provided, the hand bone is oriented from the HandPose.
 */
export function retargetHand(
  skeleton: ResolvedSkeleton,
  hand: HandPose,
  side: "left" | "right",
  handParentWorldRot: Quaternion,
  handWorldRot?: Quaternion
): RetargetedHand {
  const boneRotations = new Map<string, Quaternion>();

  // --- Step 1: Orient the hand bone itself ---
  const handBone = skeleton.getBone(`${side}Hand` as CanonicalRole);
  if (!handBone) {
    return { boneRotations, handWorldRot: new Quaternion() };
  }

  // PHASE 32 — orthonormalize the palm frame before building the wrist basis.
  //
  // Two measured defects, both in this block:
  //
  // 1. HANDEDNESS. `palmSide` was `cross(palmNormal, palmDir)`, which is the
  //    mirror of the correct order. The wrist basis is consumed as the triple
  //    (palmDir, palmNormal, palmSide), and a look-at basis is only a valid
  //    ROTATION when that triple is right-handed (X . (Y x Z) = +1). Measured
  //    over 266 real hand samples:
  //        cross(pn, pd)  [was]  right-handed   0/266   left-handed 143/266
  //        cross(pd, pn)  [now]  right-handed 143/266   left-handed   0/266
  //    The old order was never right-handed, so the wrist was framed as a
  //    REFLECTION -- the hand reads as wrenched rather than merely mis-aimed.
  //
  // 2. ORTHOGONALITY. palmDir and palmNormal are MediaPipe estimates, not an
  //    orthonormal pair: measured |palmDir . palmNormal| is 0.4253 on average
  //    and reaches 1.0000 (fully parallel) on some frames, which also makes
  //    123/266 samples degenerate for the cross product. Feeding a skewed pair
  //    into a look-at basis silently mis-frames the wrist, so palmNormal is now
  //    projected perpendicular to palmDir before use. This is also what makes
  //    the degenerate count fall to zero, because a true perpendicular pair
  //    always yields a non-zero cross product.
  //
  // `palmSide` was previously computed and then FREED WITHOUT EVER BEING USED
  // (it appeared only in the free list), so no existing behaviour depended on
  // the old value. It is now the third axis of the basis and is actually used.
  const palmDir = cvec(hand.palmDir, _palmDir).normalize();
  // Re-orthogonalize the palm normal against the palm forward axis.
  const palmNormal = cvec(hand.palmNormal, _palmNormal)
    .projectOnPlane(palmDir)
    .normalize();
  if (palmNormal.lengthSq() < 1e-12) {
    // palmNormal was parallel to palmDir: fall back to any axis not parallel to
    // the palm forward direction, so the basis stays well defined.
    palmNormal.copy(_axisZ).projectOnPlane(palmDir);
    if (palmNormal.lengthSq() < 1e-12) {
      palmNormal.copy(CANONICAL_UP).projectOnPlane(palmDir);
    }
    palmNormal.normalize();
  }
  // Right-handed third axis: X x Y, with X = palmDir and Y = palmNormal.
  const palmSide = tmpVec().crossVectors(palmDir, palmNormal).normalize();

  // The bind-pose hand direction (from parent to hand in T-pose)
  const bindDir = _handBindDir.copy(handBone.localDir).normalize();

  if (bindDir.length() < 1e-6) {
    // Zero-length bone (e.g. no children) — can't orient, inherit parent rotation
    return { boneRotations, handWorldRot: handParentWorldRot.clone() };
  }

  // Build target basis: palmDir (forward/X), palmNormal (up/Y), palmSide (lateral/Z)
  // The hand bone's bind direction is +X (left) or -X (right)
  // We need to rotate bindDir → palmDir, with twist from palmNormal
  //
  // PHASE 32: the twist reference is now `palmSide` -- the third axis of the
  // orthonormal palm frame computed above -- instead of the previous global
  // `cross(CANONICAL_UP, bindDir)`. A GLOBAL world-up reference is meaningless
  // for a hand that points in any other direction; measured on real captures,
  // palmDir sits ~50 deg from world up on average, so the old reference was
  // effectively arbitrary and varied with world orientation. `palmSide` is
  // palm-local, handedness-correct, and continuous as the wrist rotates.
  const refSide = tmpVec().copy(palmSide);
  if (refSide.length() < 0.1) {
    refSide.crossVectors(CANONICAL_UP, bindDir).normalize();
    if (refSide.length() < 0.1) {
      refSide.crossVectors(_axisZ, bindDir).normalize();
    }
  }

  const dirQuat = tmpQuat();
  rotationBetween(bindDir, palmDir, dirQuat);
  const rotatedSide = tmpVec().copy(refSide).applyQuaternion(dirQuat);
  const twistAngle = signedAngle(rotatedSide, palmNormal, palmDir);
  const twistQuat = tmpQuat().setFromAxisAngle(palmDir, twistAngle);

  // Hand world rotation
  if (!handWorldRot) {
    // Same composition order as the body retargeter: twist applied after the aim.
    handWorldRot = tmpQuat().copy(twistQuat).multiply(dirQuat);
  } else {
    // Use provided hand rotation but adjust twist
    handWorldRot = tmpQuat().copy(handWorldRot);
  }

  const handLocalRot = tmpQuat()
    .copy(handParentWorldRot)
    .invert()
    .multiply(handWorldRot);

  boneRotations.set(handBone.name, handLocalRot.clone());
  freeVec(palmSide);
  freeVec(refSide);
  freeVec(rotatedSide);
  freeQuat(dirQuat);
  freeQuat(twistQuat);
  freeQuat(handLocalRot);

  // --- Step 2: Retarget each finger ---
  const fingerBones = getFingerBoneNames(skeleton, side);

  for (const fingerName of ALL_FINGERS) {
    const bones = fingerBones[fingerName];
    if (bones.length === 0) continue;

    const fingerPose = hand.fingers[fingerName];
    if (!fingerPose) continue;

    retargetSingleFinger(
      skeleton,
      bones,
      fingerPose,
      fingerName,
      side,
      handWorldRot,
      boneRotations,
      // PHASE 25: the palm frame is REQUIRED for a correct finger hinge. The
      // previous version derived its flex axis from the GLOBAL world up, which
      // is only a valid proxy when the palm happens to point straight up.
      // Measured on the real captures, palmDir sits 49.9 deg from world up on
      // average (worst case 90 deg), so that proxy put the hinge axis 22.6 deg
      // out of the palm plane on average and up to 86.5 deg -- which curls
      // fingers sideways out of the palm instead of into it.
      palmDir,
      palmNormal
    );
  }

  // Return handWorldRot without freeing it (caller uses it)
  return {
    boneRotations,
    handWorldRot: handWorldRot.clone(),
  };
}

/**
 * Retarget a single finger (1-4 phalanges) from canonical finger pose.
 *
 * The finger's root (first phalange) is attached to the hand bone.
 * Each phalange bone gets:
 * - Direction aligned to the finger direction (rotated from bind pose)
 * - Curl (flexion) applied at each joint
 *
 * For the thumb, the curl axis is different (it bends in the opposite plane).
 */
/**
 * PHASE 30 — `FLEXION_DISTRIBUTION` and `flexionWeights()` are REMOVED.
 *
 * They existed only to divide one total cap across a chain, e.g. give the tail
 * joint a small share. With `JOINT_FLEX_MAX` each joint now carries its own
 * anatomically-derived ceiling, so a distribution table would be a second,
 * conflicting source of truth. See that constant for the measured comparison.
 *
 * The compounding behaviour these weights sat next to is unchanged and still
 * required: a bone flexes only if it has a child (the joint that swings the NEXT
 * phalanx), and each phalanx continues from where the previous one ended.
 */
function retargetSingleFinger(
  skeleton: ResolvedSkeleton,
  boneNames: string[],
  fingerPose: FingerPose,
  fingerName: FingerName,
  _side: "left" | "right",
  handWorldRot: Quaternion,
  boneRotations: Map<string, Quaternion>,
  // PHASE 25: palm-local frame. A finger hinges about an axis that lies IN the
  // palm plane, so the axis must be derived from `palmNormal` (perpendicular to
  // that plane) rather than from the global world up. See the call site for the
  // measured justification.
  palmDir: Vector3,
  palmNormal: Vector3
) {
  const fingerDir = cvec(fingerPose.dir, _fingerDir).normalize();
  const curl = Math.max(0, Math.min(1, fingerPose.curl));

  // PHASE 30: per-joint anatomical ceiling instead of one total cap.
  const jointMax = JOINT_FLEX_MAX[fingerName] ?? [90, 110, 80];

  // PHASE 25 — palm-relative flex reference.
  //
  // Was: `cross(fingerDir, CANONICAL_UP)` — a GLOBAL reference that silently
  // assumed the hand points up. The palm normal is already available and is the
  // anatomically correct reference: flexion curls the finger TOWARD the palm,
  // i.e. within the palm plane, so the hinge axis is perpendicular to the palm
  // normal. Projecting the palm normal perpendicular to the finger direction
  // yields that axis for ANY hand orientation, and it is continuous as the hand
  // rotates, so there is no world-frame singularity to flip through.
  const fingerSide = tmpVec()
    .copy(palmNormal)
    .projectOnPlane(fingerDir)
    .normalize();
  if (fingerSide.length() < 0.1) {
    // Finger is parallel to the palm normal (pointing straight out of the
    // palm): fall back to the palm's own forward direction, still palm-local.
    fingerSide.copy(palmDir).projectOnPlane(fingerDir).normalize();
    if (fingerSide.length() < 0.1) {
      fingerSide.copy(palmDir);
    }
  }

  // Accumulated local rotation from parent chain: handWorldRot * accumulated
  // gives the world rotation of the current bone's parent
  const accumulatedLocal = tmpQuat().identity();

  // PHASE 6 — compound flexion (behaviour unchanged, ceiling changed).
  // A bone only flexes if it has a child (the joint that moves the NEXT
  // phalanx). Xbot's 4th phalanx is a leaf, so it carries no flexion of its
  // own; the joint that swings it is the one before it.
  const jointBones: number[] = [];
  for (let i = 0; i < boneNames.length; i++) {
    const b = skeleton.bones.get(boneNames[i]);
    if (b && b.children.length > 0) jointBones.push(i);
  }

  // Running finger direction, compounded joint by joint. This is the key change:
  // each phalanx continues from where the PREVIOUS one ended, so rotations
  // accumulate instead of every bone aiming at the same un-curled direction.
  const dirSoFar = tmpVec().copy(fingerDir);

  for (let i = 0; i < boneNames.length; i++) {
    const bone = skeleton.bones.get(boneNames[i]);
    if (!bone) continue;

    // PHASE 26 — leaf phalanges must inherit the parent's bind direction.
    //
    // `bone.localDir` is defined by the resolver as (child - this bone) in local
    // space, so a LEAF bone (no children) has localDir === (0,0,0). Measured on
    // Xbot: 10 of 40 finger bones are in that state -- the 4th phalanx of every
    // finger on both hands (Index4, Middle4, Ring4, Pinky4, Thumb4).
    //
    // The old code normalized that zero vector and, finding length < 1e-6,
    // assigned IDENTITY and `continue`d. So the most distal bone of every
    // finger -- the visible fingertip -- never aimed at anything, and the TIP
    // landmark contributed nothing. That is a direct cause of the reported
    // clawing/splaying: the last segment is rigid and the chain above it is
    // being asked to absorb the whole curl.
    //
    // The fix uses the PARENT bone's localDir as the stand-in. For a relaxed
    // hand the last two phalanges are near-collinear, so this is the correct
    // direction, and it keeps the chain coherent: measured dot products between
    // consecutive effective directions are all >= 0.99 (0 discontinuities at a
    // 0.5 threshold) across all five left-hand fingers.
    const bindDir = _hBindDir.copy(bone.localDir).normalize();
    if (bindDir.length() < 1e-6) {
      const parentBone = bone.parent ? skeleton.bones.get(bone.parent) : undefined;
      if (parentBone && parentBone.localDir.lengthSq() > 1e-12) {
        bindDir.copy(parentBone.localDir).normalize();
      }
    }
    if (bindDir.length() < 1e-6) {
      const localRot = new Quaternion();
      boneRotations.set(bone.name, localRot);
      continue;
    }

    // PHASE 30: each joint flexes toward ITS OWN anatomical maximum, scaled by
    // curl. `jointBones` is ordered proximal->distal, so index 0 is the MCP, 1 the
    // PIP and 2 the DIP. A chain with more joints than JOINT_FLEX_MAX entries
    // reuses the last (most distal) value, which keeps a long chain in range
    // instead of dropping to zero.
    const jointIdx = jointBones.indexOf(i);
    const maxDeg = jointMax[Math.min(jointIdx, jointMax.length - 1)] ?? 80;
    const flexAngle = jointIdx >= 0 ? (curl * maxDeg * Math.PI) / 180 : 0;

    // Flexion axis: perpendicular to the CURRENT direction, in the finger's
    // bend plane. Recomputed per joint because the direction has rotated.
    let flexAxis = tmpVec().crossVectors(dirSoFar, fingerSide).normalize();
    if (flexAxis.length() < 0.1) {
      // PHASE 25: fall back to the PALM direction, not the global world up.
      // The previous fallback re-introduced exactly the world-frame assumption
      // this fix removes, and only on the degenerate path.
      flexAxis = tmpVec().copy(palmDir).projectOnPlane(dirSoFar).normalize();
      if (flexAxis.length() < 0.1) {
        flexAxis.copy(palmDir);
      }
    }

    const flexQuat = tmpQuat().setFromAxisAngle(flexAxis, flexAngle);
    // Compound: rotate the running direction, don't restart from fingerDir.
    dirSoFar.applyQuaternion(flexQuat).normalize();

    const targetDir = tmpVec().copy(dirSoFar);
    // The target side vector: rotated fingerSide by the same flexion
    const targetSide = tmpVec().copy(fingerSide).applyQuaternion(flexQuat);

    // Parent world rotation = handWorldRot * accumulated local rotations
    const parentWorldRot = tmpQuat().copy(handWorldRot).multiply(accumulatedLocal);

    const localRot = retargetBoneDirection(
      bone,
      targetDir,
      targetSide,
      parentWorldRot
    );

    boneRotations.set(bone.name, localRot);

    // Accumulate: accumulatedLocal *= localRot
    accumulatedLocal.multiply(localRot);

    freeVec(targetDir);
    freeVec(targetSide);
    freeVec(flexAxis);
    freeQuat(flexQuat);
    freeQuat(parentWorldRot);
    // PHASE 5: `localRot` is NOT freed here. It was just stored in
    // `boneRotations`, which is handed back to the caller and retained for the
    // lifetime of the frame — returning it to the pool let the very next
    // tmpQuat() pop it and overwrite it in place (setFromAxisAngle /
    // copy().multiply()), silently rewriting the rotation we had just stored.
    // That corrupted every finger bone except the last one written per chain,
    // which is why the thumb and pinky (and sporadically the index/middle)
    // came out tens of degrees off axis while the hand bone itself was exact.
    // The pool object inside retargetBoneDirection is intentionally left
    // un-freed too; retargetBoneDirection already returns a .clone().
  }
  freeVec(fingerSide);
  freeVec(dirSoFar);
  freeQuat(accumulatedLocal);
}

/**
 * Apply the full hand pose to both hands, integrating with the body retargeter's
 * bone rotation map. The hand world rotation is computed from the forearm chain.
 */
export function retargetBothHands(
  skeleton: ResolvedSkeleton,
  leftHand: HandPose,
  rightHand: HandPose,
  forearmWorldRotations: Map<string, Quaternion>
): Map<string, Quaternion> {
  const boneRotations = new Map<string, Quaternion>();

  // Left hand
  const leftForearmBone = skeleton.getBone("leftForearm" as CanonicalRole);
  const leftForearmRot = leftForearmBone
    && forearmWorldRotations.has(leftForearmBone.name)
    ? (forearmWorldRotations.get(leftForearmBone.name) as Quaternion)
    : _identH.identity();
  const leftResult = retargetHand(
    skeleton,
    leftHand,
    "left",
    leftForearmRot
  );
  for (const [name, rot] of leftResult.boneRotations) {
    boneRotations.set(name, rot);
  }

  // Right hand
  const rightForearmBone = skeleton.getBone("rightForearm" as CanonicalRole);
  // PHASE 11: `_identH` replaces the two `new Quaternion()` fallbacks. It is a
  // module singleton, and the value is only read (inverted/multiplied into a
  // pooled temporary inside retargetHand) before the next use, so sharing it is
  // safe. The original allocated one per hand, per inference.
  const rightForearmRot = rightForearmBone
    && forearmWorldRotations.has(rightForearmBone.name)
    ? (forearmWorldRotations.get(rightForearmBone.name) as Quaternion)
    : _identH.identity();
  const rightResult = retargetHand(
    skeleton,
    rightHand,
    "right",
    rightForearmRot
  );
  for (const [name, rot] of rightResult.boneRotations) {
    boneRotations.set(name, rot);
  }

  return boneRotations;
}
