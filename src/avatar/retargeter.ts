/**
 * Main retargeter: CanonicalPose → avatar bone rotations.
 *
 * This is the central entry point for the new retargeting pipeline.
 * It takes a canonical pose (from any source: MediaPipe, SMPL-X, etc.)
 * and a resolved skeleton (from the GLB) and produces local-space bone
 * rotations for every bone in the skeleton.
 *
 * Architecture:
 *   CanonicalPose → [retargetBody] → body bone rotations
 *   CanonicalPose → [retargetHand] → hand + finger bone rotations
 *   Combined → AvatarBoneMap (bone name → Quaternion)
 *
 * This replaces both retarget.ts (quaternion path) and
 * smplxConverter.ts (SMPL-X path) with a single, consistent pipeline.
 */
import { Quaternion, Vector3 } from "three";
import { CanonicalPose } from "./canonicalPose";
import { ResolvedSkeleton } from "./boneResolver";
import { retargetBody } from "./bodyRetargeter";
import { retargetBothHands } from "./handRetargeter";

/** A complete set of bone rotations for the avatar, keyed by bone name. */
export type AvatarBoneMap = Map<string, Quaternion>;

export interface RetargetedPose {
  /** Local-space bone rotations keyed by avatar bone name. */
  boneRotations: AvatarBoneMap;
  /** World position of the root (hips) joint. */
  hipsWorldPos: Vector3;
  /** World-space rotations for all bones (for child lookups). */
  worldRotations: Map<string, Quaternion>;
}

/**
 * Retarget a full canonical pose to avatar bone rotations.
 *
 * This produces rotations for all body bones AND all finger bones.
 * The result can be applied directly to the avatar's skeleton via
 * `bone.quaternion.copy(rotation)`.
 */
export function retarget(
  skeleton: ResolvedSkeleton,
  pose: CanonicalPose
): RetargetedPose {
  const boneRotations = new Map<string, Quaternion>();
  const worldRotations = new Map<string, Quaternion>();

  // --- Step 1: Retarget body (spine, head, arms, legs) ---
  const bodyResult = retargetBody(skeleton, pose.body);

  // Merge body bone rotations
  for (const [name, rot] of bodyResult.boneRotations) {
    boneRotations.set(name, rot);
  }
  // Merge world rotations
  for (const [name, rot] of bodyResult.worldRotations) {
    worldRotations.set(name, rot);
  }

  // --- Step 2: Retarget hands and fingers ---
  // The hand retargeter uses the world rotations of forearm bones
  // to properly orient the hands relative to the arm chain.
  const handRotations = retargetBothHands(
    skeleton,
    pose.leftHand,
    pose.rightHand,
    worldRotations
  );

  // Merge hand bone rotations
  for (const [name, rot] of handRotations) {
    boneRotations.set(name, rot);
  }

  return {
    boneRotations,
    hipsWorldPos: bodyResult.hipsWorldPos,
    worldRotations,
  };
}

/**
 * Compute world-space rotations from local-space bone rotations.
 * Traverses the skeleton hierarchy from root to leaves.
 *
 * This is needed because the hand retargeter requires the forearm's
 * world rotation (not local) to correctly orient the hands.
 */
/**
 * Convert the old AvatarFrame format (tuple-based quaternions) to
 * the new AvatarBoneMap (Three.js Quaternion objects).
 *
 * This allows backward compatibility with existing animation data
 * while using the new skeleton resolution system.
 */
export function avatarBoneMapToThree(
  boneMap: AvatarBoneMap,
): AvatarBoneMap {
  return boneMap;
}

/**
 * Apply retargeted bone rotations to a Three.js skeleton.
 *
 * @param boneRotations The rotations from retarget()
 * @param skeleton The resolved skeleton (for bone references)
 */
export function applyBoneRotations(
  boneRotations: AvatarBoneMap,
  skeleton: ResolvedSkeleton
): void {
  for (const [boneName, rotation] of boneRotations) {
    const bone = skeleton.bones.get(boneName);
    if (bone && bone.bone) {
      bone.bone.quaternion.copy(rotation);
    }
  }
}
