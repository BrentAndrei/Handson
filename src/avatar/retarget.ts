const POSE_COUNT = 33;
const HAND_COUNT = 21;

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export type ParentWorldQuats = Record<string, Quat>;


export interface AvatarFrame {
  root: [number, number, number];
  hips: [number, number, number];
  torso: Quat;
  head: Quat;
  neck: Quat;
  leftShoulder: Quat;
  leftUpperArm: Quat;
  leftForearm: Quat;
  leftHand: Quat;
  rightShoulder: Quat;
  rightUpperArm: Quat;
  rightForearm: Quat;
  rightHand: Quat;
}

function vSub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function vLen(v: [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function getData(data: Float32Array, offset: number, index: number): [number, number, number] {
  const base = offset + index * 3;
  return [data[base], data[base + 1], data[base + 2]];
}

const QUAT_IDENTITY: Quat = [0, 0, 0, 1];

const DEFAULT_BONE_DIRS: Record<string, Vec3> = {
  head: [0, 0, 1],
  neck: [0, 0, 1],
  torso: [0, 1, 0],
  leftShoulder: [0.378, 0.866, -0.213],
  rightShoulder: [-0.378, 0.866, -0.213],
  leftUpperArm: [1, 0, 0],
  rightUpperArm: [-1, 0, 0],
  leftForearm: [1, 0, 0],
  rightForearm: [-1, 0, 0],
  leftHand: [1, 0, 0],
  rightHand: [-1, 0, 0],
};

function normalizeVec3(v: Vec3): Vec3 {
  const len = vLen(v);
  if (len < 1e-8) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function mediapipeToGltf(v: Vec3): Vec3 {
  return [v[0], -v[1], -v[2]];
}

const Z_GAIN = 0.6;
const Z_CLAMP = 0.5;
function dampZ(v: Vec3): Vec3 {
  return [v[0], v[1], clamp(v[2] * Z_GAIN, -Z_CLAMP, Z_CLAMP)];
}

function inverseRotateByQuat(v: Vec3, q: Quat): Vec3 {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  const x2 = qx * qx;
  const y2 = qy * qy;
  const z2 = qz * qz;
  const m11 = 1 - 2 * (y2 + z2);
  const m12 = 2 * (qx * qy + qz * qw);
  const m13 = 2 * (qx * qz - qy * qw);
  const m21 = 2 * (qx * qy - qz * qw);
  const m22 = 1 - 2 * (x2 + z2);
  const m23 = 2 * (qy * qz + qx * qw);
  const m31 = 2 * (qx * qz + qy * qw);
  const m32 = 2 * (qy * qz - qx * qw);
  const m33 = 1 - 2 * (x2 + y2);
  return [
    m11 * vx + m12 * vy + m13 * vz,
    m21 * vx + m22 * vy + m23 * vz,
    m31 * vx + m32 * vy + m33 * vz,
  ];
}

function rotateByQuat(v: Vec3, q: Quat): Vec3 {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  const x2 = qx * qx;
  const y2 = qy * qy;
  const z2 = qz * qz;
  const m11 = 1 - 2 * (y2 + z2);
  const m12 = 2 * (qx * qy - qz * qw);
  const m13 = 2 * (qx * qz + qy * qw);
  const m21 = 2 * (qx * qy + qz * qw);
  const m22 = 1 - 2 * (x2 + z2);
  const m23 = 2 * (qy * qz - qx * qw);
  const m31 = 2 * (qx * qz - qy * qw);
  const m32 = 2 * (qy * qz + qx * qw);
  const m33 = 1 - 2 * (x2 + y2);
  return [
    m11 * vx + m12 * vy + m13 * vz,
    m21 * vx + m22 * vy + m23 * vz,
    m31 * vx + m32 * vy + m33 * vz,
  ];
}

function multiplyQuats(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz
  ];
}

function alignBoneToDirection(
  defaultBoneDir: Vec3,
  targetDir: Vec3,
  parentWorldQuat?: Quat,
  upVector?: Vec3,
): Quat {
  // Convert target from MediaPipe (Y-down, Z-inward) to GLTF (Y-up, Z-outward)
  const gltfTarget = mediapipeToGltf(targetDir);
  // Transform from GLTF world space into parent-local space
  const localTarget = parentWorldQuat ? inverseRotateByQuat(gltfTarget, parentWorldQuat) : gltfTarget;
  const target = normalizeVec3(localTarget);
  if (target[0] === 0 && target[1] === 0 && target[2] === 0) return [...QUAT_IDENTITY];

  const dx = defaultBoneDir[0];
  const dy = defaultBoneDir[1];
  const dz = defaultBoneDir[2];
  const tx = target[0];
  const ty = target[1];
  const tz = target[2];

  const dot = dx * tx + dy * ty + dz * tz;

  if (dot < -0.99999) {
    const ortho = Math.abs(dy) > 0.9 ? [1, 0, 0] : [0, 0, 1];
    const axis = [
      dy * ortho[2] - dz * ortho[1],
      dz * ortho[0] - dx * ortho[2],
      dx * ortho[1] - dy * ortho[0],
    ];
    const len = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
    if (len < 1e-8) return [...QUAT_IDENTITY];
    return [axis[0] / len, axis[1] / len, axis[2] / len, 0];
  }

  const s = Math.sqrt((1 + dot) * 2);
  const invS = s < 1e-8 ? 0 : 1 / s;
  const cx = dy * tz - dz * ty;
  const cy = dz * tx - dx * tz;
  const cz = dx * ty - dy * tx;

  const shortestArc: Quat = [cx * invS, cy * invS, cz * invS, s * 0.5];

  // If an upVector is provided, use it to constrain the twist (roll) around
  // the bone's long axis. This is essential for anatomically correct arm/hand
  // orientation — without it, the shortest-arc solution leaves roll undefined,
  // which causes arbitrary palm-up/down postures.
  if (upVector) {
    const localUp = parentWorldQuat ? inverseRotateByQuat(mediapipeToGltf(upVector), parentWorldQuat) : mediapipeToGltf(upVector);
    const localUpN = normalizeVec3(localUp);
    if (localUpN[0] !== 0 || localUpN[1] !== 0 || localUpN[2] !== 0) {
      // Transform the default bone up into the target frame's local space
      // by applying the shortest-arc rotation
      const rotatedDefaultUp = rotateByQuat(defaultBoneDir, shortestArc);
      const rotatedDefaultUpN = normalizeVec3(rotatedDefaultUp);
      
      // Compute the twist needed to align the rotated default up with the target up
      const twistAxis = normalizeVec3(target);
      
      // Calculate the remaining angle between projected up vectors
      const dotProj = rotatedDefaultUpN[0] * twistAxis[0] + rotatedDefaultUpN[1] * twistAxis[1] + rotatedDefaultUpN[2] * twistAxis[2];
      const projDefault: Vec3 = [
        rotatedDefaultUpN[0] - twistAxis[0] * dotProj,
        rotatedDefaultUpN[1] - twistAxis[1] * dotProj,
        rotatedDefaultUpN[2] - twistAxis[2] * dotProj,
      ];
      const dotLocalUp = localUpN[0] * twistAxis[0] + localUpN[1] * twistAxis[1] + localUpN[2] * twistAxis[2];
      const projTarget: Vec3 = [
        localUpN[0] - twistAxis[0] * dotLocalUp,
        localUpN[1] - twistAxis[1] * dotLocalUp,
        localUpN[2] - twistAxis[2] * dotLocalUp,
      ];
      
      const projDefaultLen = vLen(projDefault);
      const projTargetLen = vLen(projTarget);
      
      if (projDefaultLen > 1e-8 && projTargetLen > 1e-8) {
        const twistCos = (projDefault[0] * projTarget[0] + projDefault[1] * projTarget[1] + projDefault[2] * projTarget[2]) / (projDefaultLen * projTargetLen);
        const twistSin = (twistAxis[0] * (projDefault[1] * projTarget[2] - projDefault[2] * projTarget[1]) +
                          twistAxis[1] * (projDefault[2] * projTarget[0] - projDefault[0] * projTarget[2]) +
                          twistAxis[2] * (projDefault[0] * projTarget[1] - projDefault[1] * projTarget[0])) / (projDefaultLen * projTargetLen);
        
        const halfAngle = 0.5 * Math.atan2(twistSin, twistCos);
        const twistW = Math.cos(halfAngle);
        const twistS = Math.sin(halfAngle);
        const twistQuat: Quat = [
          twistAxis[0] * twistS,
          twistAxis[1] * twistS,
          twistAxis[2] * twistS,
          twistW
        ];
        
        // Apply twist after the shortest-arc rotation
        return multiplyQuats(shortestArc, twistQuat);
      }
    }
  }

  return shortestArc;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

// === Spline-based quaternion interpolation ===

// Logarithmic map: convert quaternion to rotation vector (axis * angle)
function logQuat(q: Quat): Vec3 {
  const [x, y, z, w] = q;
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len < 1e-10) return [0, 0, 0];
  const angle = 2 * Math.atan2(len, w);
  const s = angle / len;
  return [x * s, y * s, z * s];
}

// Exponential map: convert rotation vector back to quaternion
function expQuat(v: Vec3): Quat {
  const [x, y, z] = v;
  const angle = Math.sqrt(x * x + y * y + z * z);
  if (angle < 1e-10) return [...QUAT_IDENTITY];
  const s = Math.sin(angle / 2) / angle;
  return [x * s, y * s, z * s, Math.cos(angle / 2)];
}

// Quaternion dot product
function quatDot(a: Quat, b: Quat): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

// Align quaternion b to same hemisphere as a (for shortest path)
function alignQuat(a: Quat, b: Quat): Quat {
  if (quatDot(a, b) < 0) {
    return [-b[0], -b[1], -b[2], -b[3]];
  }
  return [...b];
}

function multiplyInverse(a: Quat, b: Quat): Quat {
  const bConj: Quat = [-b[0], -b[1], -b[2], b[3]];
  return multiplyQuats(a, bConj);
}

// Spherical linear interpolation (SLERP) between quaternions
function slerpQuat(a: Quat, b: Quat, t: number): Quat {
  const dot = quatDot(a, b);
  const absDot = Math.abs(dot);
  
  if (absDot > 0.9995) {
    // Near-linear interpolation for very close quaternions
    const sign = dot < 0 ? -1 : 1;
    const result: Quat = [
      a[0] + t * (b[0] * sign - a[0]),
      a[1] + t * (b[1] * sign - a[1]),
      a[2] + t * (b[2] * sign - a[2]),
      a[3] + t * (b[3] * sign - a[3])
    ];
    const len = Math.sqrt(result[0]**2 + result[1]**2 + result[2]**2 + result[3]**2);
    return [result[0]/len, result[1]/len, result[2]/len, result[3]/len];
  }
  
  const theta = Math.acos(absDot);
  const sinTheta = Math.sin(theta);
  const scale0 = Math.sin((1 - t) * theta) / sinTheta;
  const scale1 = Math.sin(t * theta) / sinTheta;
  
  const sign = dot < 0 ? -1 : 1;
  
  return [
    a[0] * scale0 + b[0] * scale1 * sign,
    a[1] * scale0 + b[1] * scale1 * sign,
    a[2] * scale0 + b[2] * scale1 * sign,
    a[3] * scale0 + b[3] * scale1 * sign
  ];
}

// Proper Squad (Spherical Quadrangle Interpolation) for smooth C1-continuous quaternion curves
export function squad(q0: Quat, q1: Quat, q2: Quat, q3: Quat, t: number): Quat {
  // Ensure shortest path between all quaternions
  if (quatDot(q0, q1) < 0) q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
  if (quatDot(q1, q2) < 0) q2 = [-q2[0], -q2[1], -q2[2], -q2[3]];
  if (quatDot(q2, q3) < 0) q3 = [-q3[0], -q3[1], -q3[2], -q3[3]];
  
  // Compute intermediate control points using logarithmic maps
  // s1 = q1 * exp(-0.25 * (ln(q1^-1 * q0) + ln(q1^-1 * q2)))
  const ln1 = logQuat(multiplyInverse(q1, q0));
  const ln2 = logQuat(multiplyInverse(q1, q2));
  const s1Vec: Vec3 = [(ln2[0] + ln1[0]) * -0.25, (ln2[1] + ln1[1]) * -0.25, (ln2[2] + ln1[2]) * -0.25];
  const s1 = multiplyQuats(q1, expQuat(s1Vec));
  
  // s2 = q2 * exp(-0.25 * (ln(q2^-1 * q1) + ln(q2^-1 * q3)))
  const ln3 = logQuat(multiplyInverse(q2, q1));
  const ln4 = logQuat(multiplyInverse(q2, q3));
  const s2Vec: Vec3 = [(ln4[0] + ln3[0]) * -0.25, (ln4[1] + ln3[1]) * -0.25, (ln4[2] + ln3[2]) * -0.25];
  const s2 = multiplyQuats(q2, expQuat(s2Vec));
  
  // Ensure s1, s2 are on the same hemisphere
  if (quatDot(s1, s2) < 0) s2[0] = -s2[0], s2[1] = -s2[1], s2[2] = -s2[2], s2[3] = -s2[3];
  
  return slerpQuat(s1, s2, smoothstep(t));
}

// Enhanced spline-based quaternion smoothing
export function smoothFramesSpline(frames: AvatarFrame[], windowSize: number = 5): AvatarFrame[] {
  if (frames.length <= 2) return smoothFrames(frames, windowSize);
  // Deep copy to avoid mutating input frames
  const copied: AvatarFrame[] = frames.map((f) => ({ ...f }));
  const smoothed: AvatarFrame[] = [];
  const keys: (keyof AvatarFrame)[] = [
    'root', 'hips', 'torso', 'head', 'neck',
    'leftShoulder', 'leftUpperArm', 'leftForearm', 'leftHand',
    'rightShoulder', 'rightUpperArm', 'rightForearm', 'rightHand',
  ];
  
  // First pass: hemisphere alignment (ensure consistent quaternion signs)
  for (const key of keys) {
    const isQuat = copied[0][key].length >= 4;
    if (isQuat) {
      for (let i = 1; i < copied.length; i++) {
        const prev = copied[i-1][key] as Quat;
        const curr = copied[i][key] as Quat;
        if (quatDot(prev, curr) < 0) {
          copied[i][key] = [-curr[0], -curr[1], -curr[2], -curr[3]] as any;
        }
      }
    }
  }
  
  // Second pass: catmull-rom style temporal smoothing using quaternion log maps
  for (let i = 0; i < copied.length; i++) {
    const out = {} as AvatarFrame;
    for (const key of keys) {
      const p0 = copied[i][key];
      const isQuat = p0.length >= 4;
      
      if (isQuat) {
        // Get neighboring keyframes (catmull-rom)
        const pPrev = copied[Math.max(0, i - 1)][key] as Quat;
        const pNext = copied[Math.min(copied.length - 1, i + 1)][key] as Quat;
        const pNext2 = copied[Math.min(copied.length - 1, i + 2)][key] as Quat;
        
        // Align neighbors to current frame
        const alignedPrev = alignQuat(p0 as Quat, pPrev);
        const alignedNext = alignQuat(p0 as Quat, pNext);
        const alignedNext2 = alignQuat(alignedNext, pNext2);
        
        // Interpolate using squad with tension
        const result = squad(alignedPrev, p0 as Quat, alignedNext, alignedNext2, 0);
        (out as any)[key] = result;
      } else {
        // Linear smoothing for position vectors
        let sx = 0, sy = 0, sz = 0;
        let count = 0;
        const half = Math.floor(windowSize / 2);
        for (let j = -half; j <= half; j++) {
          const idx = Math.min(Math.max(i + j, 0), copied.length - 1);
          const p = copied[idx][key] as number[];
          sx += p[0]; sy += p[1]; sz += p[2];
          count++;
        }
        (out as any)[key] = [sx / count, sy / count, sz / count];
      }
    }
    smoothed.push(out);
  }
  return smoothed;
}

export function retargetFrame(
  data: Float32Array,
  frameIndex: number,
  parentWorldQuats?: ParentWorldQuats,
): AvatarFrame | null {
  const frameOffset = frameIndex * 345;
  if (frameOffset + 345 > data.length) return null;

  const poseOffset = frameOffset;
  const leftHandOffset = frameOffset + POSE_COUNT * 3;
  const rightHandOffset = frameOffset + (POSE_COUNT + HAND_COUNT) * 3;

  const nose = getData(data, poseOffset, 0);
  const leftShoulder = getData(data, poseOffset, 11);
  const rightShoulder = getData(data, poseOffset, 12);
  const leftElbow = getData(data, poseOffset, 13);
  const rightElbow = getData(data, poseOffset, 14);
  const leftWrist = getData(data, poseOffset, 15);
  const rightWrist = getData(data, poseOffset, 16);

  const shoulderMid: Vec3 = [
    (leftShoulder[0] + rightShoulder[0]) / 2,
    (leftShoulder[1] + rightShoulder[1]) / 2,
    (leftShoulder[2] + rightShoulder[2]) / 2,
  ];

  // Torso forward: from shoulder midpoint toward nose, defines spine tilt
  const torsoForward: Vec3 = dampZ(vSub(nose, shoulderMid));
  const root: [number, number, number] = [0, 0, 0];
  const hipsRel: [number, number, number] = [0, 0, 0];

  // Compute torso (spine) rotation: align spine default direction (up [0,1,0])
  // to the computed forward direction, with shoulder line as up reference
  const shoulderLine: Vec3 = dampZ(vSub(rightShoulder, leftShoulder));
  const torsoRot = alignBoneToDirection(
    DEFAULT_BONE_DIRS.torso,
    torsoForward,
    parentWorldQuats?.["root"],
    shoulderLine,
  );

  const neckDir: Vec3 = dampZ(nose) as Vec3;
  const headDir: Vec3 = dampZ([nose[0], nose[1] + 0.15, nose[2]]) as Vec3;

  const leftHand = getData(data, leftHandOffset, 0);
  const rightHand = getData(data, rightHandOffset, 0);
  const leftHandMcp = getData(data, leftHandOffset, 9);
  const rightHandMcp = getData(data, rightHandOffset, 9);

  const neckRot = alignBoneToDirection(DEFAULT_BONE_DIRS.neck, neckDir, parentWorldQuats?.["torso"], [0, 0, 1]);
  const headRot = alignBoneToDirection(DEFAULT_BONE_DIRS.head, headDir, parentWorldQuats?.["neck"]);

  const shoulderDirL = dampZ(vSub(leftElbow, leftShoulder)) as [number, number, number];
  const shoulderDirR = dampZ(vSub(rightElbow, rightShoulder)) as [number, number, number];
  const leftShoulderRot = alignBoneToDirection(DEFAULT_BONE_DIRS.leftShoulder, shoulderDirL, parentWorldQuats?.["torso"]);
  const rightShoulderRot = alignBoneToDirection(DEFAULT_BONE_DIRS.rightShoulder, shoulderDirR, parentWorldQuats?.["torso"]);

  const leftElbowRel = vSub(leftElbow, leftShoulder);
  const rightElbowRel = vSub(rightElbow, rightShoulder);
  const leftUpperArmRot = alignBoneToDirection(DEFAULT_BONE_DIRS.leftUpperArm, dampZ(leftElbowRel), parentWorldQuats?.["leftShoulder"]);
  const rightUpperArmRot = alignBoneToDirection(DEFAULT_BONE_DIRS.rightUpperArm, dampZ(rightElbowRel), parentWorldQuats?.["rightShoulder"]);
  const leftForearmRel = vSub(leftWrist, leftElbow);
  const rightForearmRel = vSub(rightWrist, rightElbow);
  
  // For forearm twist, use the vector from left wrist to left index MCP projected onto
  // the plane perpendicular to the forearm direction. This defines the natural
  // palm orientation without arbitrary roll.
  const leftIndexMcp = getData(data, leftHandOffset, 5);
  const rightIndexMcp = getData(data, rightHandOffset, 5);
  const leftWristToIndex = vSub(leftIndexMcp, leftWrist);
  const rightWristToIndex = vSub(rightIndexMcp, rightWrist);
  
  // Compute the up-vector as the cross product of forearm direction and wrist-to-index direction
  const leftForearmUp: Vec3 = [
    leftForearmRel[1] * leftWristToIndex[2] - leftForearmRel[2] * leftWristToIndex[1],
    leftForearmRel[2] * leftWristToIndex[0] - leftForearmRel[0] * leftWristToIndex[2],
    leftForearmRel[0] * leftWristToIndex[1] - leftForearmRel[1] * leftWristToIndex[0]
  ];
  const rightForearmUp: Vec3 = [
    rightForearmRel[1] * rightWristToIndex[2] - rightForearmRel[2] * rightWristToIndex[1],
    rightForearmRel[2] * rightWristToIndex[0] - rightForearmRel[0] * rightWristToIndex[2],
    rightForearmRel[0] * rightWristToIndex[1] - rightForearmRel[1] * rightWristToIndex[0]
  ];
  
  const leftForearmRot = alignBoneToDirection(DEFAULT_BONE_DIRS.leftForearm, dampZ(leftForearmRel), parentWorldQuats?.["leftUpperArm"], dampZ(leftForearmUp));
  const rightForearmRot = alignBoneToDirection(DEFAULT_BONE_DIRS.rightForearm, dampZ(rightForearmRel), parentWorldQuats?.["rightUpperArm"], dampZ(rightForearmUp));

  const leftHandDirection = vSub(leftHandMcp, leftHand);
  const rightHandDirection = vSub(rightHandMcp, rightHand);
  // For hand twist, use the perpendicular from wrist to pinky MCP to define palm orientation
  const leftPinkyMcp = getData(data, leftHandOffset, 17);
  const rightPinkyMcp = getData(data, rightHandOffset, 17);
  const leftWristToPinky = vSub(leftPinkyMcp, leftHand);
  const rightWristToPinky = vSub(rightPinkyMcp, rightHand);
  
  const leftHandUp: Vec3 = [
    leftHandDirection[1] * leftWristToPinky[2] - leftHandDirection[2] * leftWristToPinky[1],
    leftHandDirection[2] * leftWristToPinky[0] - leftHandDirection[0] * leftWristToPinky[2],
    leftHandDirection[0] * leftWristToPinky[1] - leftHandDirection[1] * leftWristToPinky[0]
  ];
  const rightHandUp: Vec3 = [
    rightHandDirection[1] * rightWristToPinky[2] - rightHandDirection[2] * rightWristToPinky[1],
    rightHandDirection[2] * rightWristToPinky[0] - rightHandDirection[0] * rightWristToPinky[2],
    rightHandDirection[0] * rightWristToPinky[1] - rightHandDirection[1] * rightWristToPinky[0]
  ];
  
  const leftHandRot = alignBoneToDirection(DEFAULT_BONE_DIRS.leftHand, dampZ(leftHandDirection), parentWorldQuats?.["leftForearm"], dampZ(leftHandUp));
  const rightHandRot = alignBoneToDirection(DEFAULT_BONE_DIRS.rightHand, dampZ(rightHandDirection), parentWorldQuats?.["rightForearm"], dampZ(rightHandUp));

  return {
    root,
    hips: hipsRel,
    torso: torsoRot,
    head: headRot,
    neck: neckRot,
    leftShoulder: leftShoulderRot,
    leftUpperArm: leftUpperArmRot,
    leftForearm: leftForearmRot,
    leftHand: leftHandRot,
    rightShoulder: rightShoulderRot,
    rightUpperArm: rightUpperArmRot,
    rightForearm: rightForearmRot,
    rightHand: rightHandRot,
  };
}

export function retargetFrames(data: Float32Array, parentWorldQuats?: ParentWorldQuats): AvatarFrame[] {
  const frameCount = Math.floor(data.length / 345);
  const frames: AvatarFrame[] = [];
  for (let i = 0; i < frameCount; i++) {
    const frame = retargetFrame(data, i, parentWorldQuats);
    if (frame) frames.push(frame);
  }
  return frames;
}

export function smoothFrames(frames: AvatarFrame[], windowSize: number = 3): AvatarFrame[] {
  if (frames.length <= 1) return frames;
  if (windowSize < 1) windowSize = 1;
  const halfWindow = Math.floor(windowSize / 2);
  const smoothed: AvatarFrame[] = [];

  const keys: (keyof AvatarFrame)[] = [
    'root', 'hips', 'torso', 'head', 'neck',
    'leftShoulder', 'leftUpperArm', 'leftForearm', 'leftHand',
    'rightShoulder', 'rightUpperArm', 'rightForearm', 'rightHand',
  ];

  for (let i = 0; i < frames.length; i++) {
    const out = {} as AvatarFrame;
    for (const key of keys) {
      const p0 = frames[i][key];
      const isQuat = p0.length >= 4;
      let sx = 0, sy = 0, sz = 0;
      let sw = 0;
      let count = 0;
      let refVec: number[] | null = null;
      for (let j = -halfWindow; j <= halfWindow; j++) {
        const idx = Math.min(Math.max(i + j, 0), frames.length - 1);
       const p = frames[idx][key] as number[];
       if (isQuat) {
         if (refVec === null) {
           refVec = p;
         }
         const d = p[0]*refVec[0] + p[1]*refVec[1] + p[2]*refVec[2] + (p[3]!)*refVec[3];
         if (d < 0) {
           sx -= p[0]; sy -= p[1]; sz -= p[2]; sw -= p[3]!;
         } else {
           sx += p[0]; sy += p[1]; sz += p[2]; sw += p[3]!;
         }
       } else {
         sx += p[0]; sy += p[1]; sz += p[2];
       }
       count++;
      }
      const cx = sx / count;
      const cy = sy / count;
      const cz = sz / count;
       if (isQuat) {
        const cw = sw / count;
        const len = Math.sqrt(cx * cx + cy * cy + cz * cz + cw * cw);
        (out as any)[key] = len < 1e-8 ? [...QUAT_IDENTITY] : [cx / len, cy / len, cz / len, cw / len];
      } else {
        (out as any)[key] = [cx, cy, cz];
      }
    }
    smoothed.push(out);
  }
  return smoothed;
}

export function interpolateFrames(a: AvatarFrame, b: AvatarFrame, t: number): AvatarFrame {
  const eased = smoothstep(clamp(t, 0, 1));
  const out = {} as AvatarFrame;
  const keys: (keyof AvatarFrame)[] = [
    'root', 'hips', 'torso', 'head', 'neck',
    'leftShoulder', 'leftUpperArm', 'leftForearm', 'leftHand',
    'rightShoulder', 'rightUpperArm', 'rightForearm', 'rightHand',
  ];
   for (const key of keys) {
    const pa = a[key] as Quat;
    let pb = b[key] as Quat;
    if (pa.length >= 4 && pb.length >= 4) {
      // Use proper SLERP instead of LERP for quaternions
      const d = pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2] + pa[3] * pb[3];
      const aq: Quat = d < 0 ? [-pa[0], -pa[1], -pa[2], -pa[3]] : [...pa];
      const bq: Quat = d < 0 ? [-pb[0], -pb[1], -pb[2], -pb[3]] : [...pb];
      const result = slerpQuat(aq, bq, eased);
      (out as any)[key] = result;
    } else {
      (out as any)[key] = [
        pa[0] + (pb[0] - pa[0]) * eased,
        pa[1] + (pb[1] - pa[1]) * eased,
        pa[2] + (pb[2] - pa[2]) * eased,
      ];
    }
  }
  return out;
}
