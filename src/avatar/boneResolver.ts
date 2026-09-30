/**
 * Avatar skeleton inspector.
 *
 * Instead of hardcoding bone directions (which drift from the actual
 * GLB), this module extracts the bone hierarchy, bind-pose directions,
 * and parent/child relationships directly from the loaded THREE.SkinnedMesh.
 *
 * The retargeter consumes a `ResolvedSkeleton` so it always matches the
 * model on disk.
 */
import {
  Bone,
  Quaternion,
  Object3D,
  Vector3,
  SkinnedMesh,
} from "three";
import { decomposeMatrix } from "./canonicalPose";

export interface ResolvedBone {
  name: string;
  /** Three.js Bone object from the skeleton. */
  bone: Bone;
  /** Index in skeleton.bones array. */
  index: number;
  /** Parent bone name (null = root). */
  parent: string | null;
  /** Direct children bone names. */
  children: string[];
  /** Bone direction in avatar local space (child position - parent position), normalized. */
  localDir: Vector3;
  /** Bone length in meters (avatar local units, accounting for armature scale). */
  length: number;
  /** Local bind-pose position (relative to parent). */
  localPos: Vector3;
  /** Local bind-pose quaternion (relative to parent). */
  localRot: Quaternion;
  /** World bind-pose position of the joint (absolute). */
  worldPos: Vector3;
  /** World bind-pose quaternion of the joint (absolute). */
  worldRot: Quaternion;
  /** The avatar-local-space "forward" axis this bone represents, mapped from canonical. */
  avatarForward: Vector3;
}

export interface ResolvedSkeleton {
  bones: Map<string, ResolvedBone>;
  rootBone: string;
  boneOrder: string[];
  /** The avatar's armature scale (e.g. 0.01 for cm→m). */
  armatureScale: number;
  /** Map from canonical body part → avatar bone name(s). */
  boneMapping: BoneMapping;
  /** Helper to look up a bone by canonical role. */
  getBone(role: CanonicalRole): ResolvedBone | undefined;
  /** Get the world bind-pose position of a bone. */
  getWorldPos(role: CanonicalRole): Vector3 | undefined;
  /** Get the local bind-pose direction of a bone. */
  getLocalDir(role: CanonicalRole): Vector3 | undefined;
}

/** Roles used by the canonical pose → avatar bone mapping. */
export type CanonicalRole =
  | "hips"
  | "spine1"
  | "spine2"
  | "spine3"
  | "neck"
  | "head"
  | "leftShoulder"
  | "leftUpperArm"
  | "leftForearm"
  | "leftHand"
  | "rightShoulder"
  | "rightUpperArm"
  | "rightForearm"
  | "rightHand"
  | "leftHip"
  | "leftKnee"
  | "leftAnkle"
  | "rightHip"
  | "rightKnee"
  | "rightAnkle"
  | `leftFinger_${FingerName}_${number}`
  | `rightFinger_${FingerName}_${number}`;

export type FingerName =
  | "thumb"
  | "index"
  | "middle"
  | "ring"
  | "pinky";

export interface BoneMapping {
  [role: string]: string;
}

/**
 * Extract the skeleton hierarchy and bind-pose data from a loaded GLB scene.
 *
 * The GLB should contain a SkinnedMesh (or a Group containing one). The
 * skeleton is read from `mesh.skeleton.bones`, and the bone mappings are
 * resolved using naming conventions + hierarchy.
 */
export function resolveSkeleton(
  scene: Object3D,
  options?: {
    armatureScale?: number;
    boneMapping?: Partial<BoneMapping>;
  }
): ResolvedSkeleton {
  const skinnedMeshes: SkinnedMesh[] = [];
  scene.traverse((obj) => {
    if ((obj as SkinnedMesh).isSkinnedMesh) {
      skinnedMeshes.push(obj as SkinnedMesh);
    }
  });

  if (skinnedMeshes.length === 0) {
    throw new Error("No SkinnedMesh found in GLB scene");
  }
  const mesh = skinnedMeshes[0];
  const skeleton = mesh.skeleton;
  if (!skeleton || !skeleton.bones || skeleton.bones.length === 0) {
    throw new Error("SkinnedMesh has no skeleton or bones");
  }

  const armatureScale = options?.armatureScale ?? 0.01;

  // --- Build bone records from the skeleton ---
  const bones = new Map<string, ResolvedBone>();
  const boneByName = new Map<string, Bone>();
  const boneIndexByName = new Map<string, number>();

  skeleton.bones.forEach((bone, idx) => {
    boneIndexByName.set(bone.name, idx);
    boneByName.set(bone.name, bone);
  });

  skeleton.bones.forEach((bone, idx) => {
    const name = bone.name;
    const parent = bone.parent ? bone.parent.name : null;

    // Compute local bind-pose data from the bone's own matrix
    // bone.matrix is the local transform; bone.matrixWorld is world
    const localDecomp = decomposeMatrix(bone.matrix);

    // For world bind-pose position, we need the bone's position in
    // avatar space at rest. bone.matrixWorld gives this.
    const worldDecomp = decomposeMatrix(bone.matrixWorld);

    // Local direction = child offset or from parent
    // localPos is the position relative to parent (from local matrix)
    const localPos = localDecomp.position.clone().multiplyScalar(1); // already in armature local space

    // Determine children
    const children: string[] = [];
    skeleton.bones.forEach((otherBone) => {
      if (otherBone.parent === bone) {
        children.push(otherBone.name);
      }
    });

    // Compute local direction: the offset from this bone to its first child
    // (or its own position if root). For a bone, the "direction" is the
    // vector from its joint to its child joint in local space.
    let localDir = new Vector3(0, 0, 0);
    if (children.length > 0) {
      const childBone = boneByName.get(children[0]);
      if (childBone) {
        const childLocal = decomposeMatrix(childBone.matrix).position;
        // childLocal is the child's position relative to THIS bone
        // (since this bone is the child's parent in the skeleton hierarchy).
        // The bone direction is simply childLocal, not childLocal - localPos,
        // because localPos is this bone's position in its PARENT's space.
        localDir.copy(childLocal);
      }
    }

    const length = localDir.length();
    if (length > 1e-6) {
      localDir.normalize();
    }

    bones.set(name, {
      name,
      bone,
      index: idx,
      parent,
      children,
      localDir,
      length,
      localPos,
      localRot: localDecomp.quaternion.clone(),
      worldPos: worldDecomp.position.clone(),
      worldRot: worldDecomp.quaternion.clone(),
      avatarForward: new Vector3(), // filled below
    });
  });

  // --- Resolve bone name mappings ---
  const boneMapping = resolveBoneMapping(bones, options?.boneMapping);

  // --- Fill in avatarForward for each bone ---
  // This maps the canonical body direction (X-fwd, Y-up, Z-right) to the
  // bone's local direction, so we know which canonical axis to align with
  // the bone's localDir.
  // For simplicity here, we just store the local direction and let the
  // retargeter do the mapping. But we also set a default forward.
  for (const [_name, bone] of bones) {
    // Default: use the bone's local direction as its "forward"
    bone.avatarForward.copy(bone.localDir);
  }

  // --- Determine root bone ---
  let rootName: string | null = null;
  for (const [name, bone] of bones) {
    if (bone.parent === null || !bones.has(bone.parent)) {
      rootName = name;
      break;
    }
  }
  if (!rootName && bones.size > 0) {
    // Fallback: pick the first bone with no parent in our map
    for (const [name, bone] of bones) {
      if (bone.parent === null) {
        rootName = name;
        break;
      }
    }
  }

  // Topological order (children after parents)
  const boneOrder = topologicalSort(bones, rootName ?? "");

  return {
    bones,
    rootBone: rootName ?? "",
    boneOrder,
    armatureScale,
    boneMapping,
    getBone: (role: CanonicalRole) => {
      const boneName = boneMapping[role as string];
      return boneName ? bones.get(boneName) : undefined;
    },
  getWorldPos: (role: CanonicalRole): Vector3 | undefined => {
    const boneName = boneMapping[role as string];
    const bone = boneName ? bones.get(boneName) : undefined;
    return bone?.worldPos;
  },
  getLocalDir: (role: CanonicalRole): Vector3 | undefined => {
    const boneName = boneMapping[role as string];
    const bone = boneName ? bones.get(boneName) : undefined;
    return bone ? bone.localDir.clone() : undefined;
  },
  };
}

/**
 * Resolve which avatar bone name corresponds to each canonical role.
 * Uses naming patterns + fallback to hierarchy.
 */
function resolveBoneMapping(
  bones: Map<string, ResolvedBone>,
  override?: Partial<BoneMapping>
): BoneMapping {
  const mapping: BoneMapping = {};

  // Helper: find a bone by fuzzy name match
  const findByName = (pattern: string | RegExp): ResolvedBone | undefined => {
    for (const [name, bone] of bones) {
      const lower = name.toLowerCase();
       const stripped = lower.replace(/mixamorig[:\-_]?/g, "").replace(/[_-]/g, "");
      if (pattern instanceof RegExp) {
        if (pattern.test(lower) || pattern.test(stripped)) return bone;
      } else {
        if (lower.includes(pattern.toLowerCase()) || stripped.includes(pattern.toLowerCase().replace(/[_-]/g, ""))) {
          return bone;
        }
      }
    }
    return undefined;
  };

  // Hips / root
  let b = findByName(/^hips$|^root$|^pelvis$/);
  if (b) mapping.hips = b.name;

  // Spine chain
  const spineNames = Array.from(bones.keys()).filter((n) => {
    const l = n.toLowerCase().replace("mixamorig", "");
    return /spine/i.test(l);
  }).sort((a, b) => {
    const ai = a.toLowerCase().replace("mixamorig", "").match(/spine(\d+)/);
    const bi = b.toLowerCase().replace("mixamorig", "").match(/spine(\d+)/);
    const an = ai ? parseInt(ai[1]) : 0;
    const bn = bi ? parseInt(bi[1]) : 0;
    return an - bn;
  });
  if (spineNames.length > 0) mapping.spine1 = spineNames[0];
  if (spineNames.length > 1) mapping.spine2 = spineNames[1];
  if (spineNames.length > 2) mapping.spine3 = spineNames[2];

  // Neck
  b = findByName(/neck/i);
  if (b) mapping.neck = b.name;

  // Head
  b = findByName(/^head$/i);
  if (b) mapping.head = b.name;

  // Shoulders
  const leftShoulder = findByName(/leftshoulder|leftclavicle/i);
  const rightShoulder = findByName(/rightshoulder|rightclavicle/i);
  if (leftShoulder) mapping.leftShoulder = leftShoulder.name;
  if (rightShoulder) mapping.rightShoulder = rightShoulder.name;

  // Upper arms
  const leftUpperArm = findByName(/leftarm$|leftupperarm/i);
  const rightUpperArm = findByName(/rightarm$|rightupperarm/i);
  if (leftUpperArm) mapping.leftUpperArm = leftUpperArm.name;
  if (rightUpperArm) mapping.rightUpperArm = rightUpperArm.name;

  // Forearms
  const leftForearm = findByName(/leftforearm|leftforearm|leftelbow/i);
  const rightForearm = findByName(/rightforearm|rightforearm|rightelbow/i);
  if (leftForearm) mapping.leftForearm = leftForearm.name;
  if (rightForearm) mapping.rightForearm = rightForearm.name;

  // Hands
  const leftHand = findByName(/lefthand$|lefthandmid/i);
  const rightHand = findByName(/righthand$|righthandmid/i);
  if (leftHand) mapping.leftHand = leftHand.name;
  if (rightHand) mapping.rightHand = rightHand.name;

  // Hips / legs
  const leftHip = findByName(/leftupleg|lefthip/i);
  const rightHip = findByName(/rightupleg|righthip/i);
  if (leftHip) mapping.leftHip = leftHip.name;
  if (rightHip) mapping.rightHip = rightHip.name;

  const leftKnee = findByName(/leftleg|leftknee/i);
  const rightKnee = findByName(/rightleg|rightknee/i);
  if (leftKnee) mapping.leftKnee = leftKnee.name;
  if (rightKnee) mapping.rightKnee = rightKnee.name;

  const leftAnkle = findByName(/leftfoot|leftankle/i);
  const rightAnkle = findByName(/rightfoot|rightankle/i);
  if (leftAnkle) mapping.leftAnkle = leftAnkle.name;
  if (rightAnkle) mapping.rightAnkle = rightAnkle.name;

  // Fingers — Xbot bone names are like mixamorig:LeftHandThumb1, etc.
  const fingerBases: Array<[string, FingerName]> = [
    ["Thumb", "thumb"],
    ["Index", "index"],
    ["Middle", "middle"],
    ["Ring", "ring"],
    ["Pinky", "pinky"],
  ];

  const sides: Array<["Left", "left"] | ["Right", "right"]> = [["Left", "left"], ["Right", "right"]];

  for (const [sideUpper, sideLower] of sides) {
    for (const [fingerPart, fingerName] of fingerBases) {
      for (let i = 1; i <= 4; i++) {
        const roleKey = `${sideLower}Finger_${fingerName}_${i}` as const;
        const bone = findByName(new RegExp(`${sideUpper}Hand${fingerPart}${i}`, "i"));
        if (bone) mapping[roleKey] = bone.name;
      }
    }
  }

  // Apply user overrides
  if (override) {
    for (const [k, v] of Object.entries(override)) {
      if (v !== undefined) mapping[k] = v;
    }
  }

  return mapping;
}

function topologicalSort(
  bones: Map<string, ResolvedBone>,
  root: string
): string[] {
  const visited = new Set<string>();
  const result: string[] = [];

  const visit = (name: string) => {
    if (visited.has(name)) return;
    visited.add(name);
    const bone = bones.get(name);
    if (!bone) return;
    for (const child of bone.children) {
      visit(child);
    }
    result.push(name);
  };

  visit(root);

  // Add any remaining disconnected bones
  for (const name of bones.keys()) {
    visit(name);
  }

  return result;
}

/** Get bone mapping for a finger. Returns undefined if not mapped. */
export function getFingerBoneNames(
  skeleton: ResolvedSkeleton,
  side: "left" | "right"
): Record<FingerName, string[]> {
  const result: Record<string, string[]> = {
    thumb: [],
    index: [],
    middle: [],
    ring: [],
    pinky: [],
  };

  for (const finger of ["thumb", "index", "middle", "ring", "pinky"] as FingerName[]) {
    for (let i = 1; i <= 4; i++) {
      const role = `${side}Finger_${finger}_${i}`;
      const boneName = skeleton.boneMapping[role];
      if (boneName) {
        result[finger].push(boneName);
      }
    }
  }

  return result;
}

/**
 * Get the chain of bones from a start role to an end role (inclusive).
 * Useful for getting the arm chain: [shoulder, upperArm, forearm, hand].
 */
export function getBoneChain(
  skeleton: ResolvedSkeleton,
  startRole: CanonicalRole,
  endRole: CanonicalRole
): ResolvedBone[] {
  const chain: ResolvedBone[] = [];
  let current: string | undefined = skeleton.boneMapping[endRole as string];
  const visited = new Set<string>();

  while (current && !visited.has(current)) {
    visited.add(current);
    const bone = skeleton.bones.get(current);
    if (!bone) break;
    chain.unshift(bone);
    if (current === skeleton.boneMapping[startRole as string]) break;
    current = bone.parent ?? undefined;
  }

  return chain.reverse();
}
