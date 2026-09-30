/**
 * Joint limits for the avatar skeleton.
 *
 * These are soft constraints applied during retargeting to prevent
 * anatomically impossible poses. Values are in degrees.
 *
 * The X-axis is twist (bone roll), Y-axis is the "swing" lateral,
 * Z-axis is the "swing" vertical — aligned to the bone's local frame.
 *
 * Limits are defined per-bone as min/max in degrees. The constraint
 * is enforced by clamping the swing and twist angles after the
 * directional alignment is computed.
 */

export interface JointLimit {
  twistMin: number;
  twistMax: number;
  swing1Min: number;
  swing1Max: number;
  swing2Min: number;
  swing2Max: number;
}

/** Human-readable axis order for the constraint: [twistAxis, swing1Axis, swing2Axis] */
export const CONSTRAINT_AXIS_ORDER = {
  twist: 0,
  swing1: 1,
  swing2: 2,
} as const;

/**
 * Convert degrees to radians.
 */
export function deg2rad(d: number): number {
  return (d * Math.PI) / 180;
}

/**
 * Convert radians to degrees.
 */
export function rad2deg(r: number): number {
  return (r * 180) / Math.PI;
}

/**
 * Joint limits for key body joints. These are conservative ranges
 * that cover normal human motion without allowing dislocations.
 *
 * Convention: the bone's local +Y is "up" (pointing to the child in
 * T-pose for most bones; see boneResolver for actual directions).
 * The twist axis is the bone's longitudinal axis.
 */
export const HUMAN_JOINT_LIMITS: Record<string, JointLimit> = {
  // Hip: limited rotation in place (handled as position + tilt)
  hips: {
    twistMin: -30,
    twistMax: 30,
    swing1Min: -30,
    swing1Max: 30,
    swing2Min: -15,
    swing2Max: 15,
  },
  spine1: {
    twistMin: -25,
    twistMax: 25,
    swing1Min: -35,
    swing1Max: 40,
    swing2Min: -25,
    swing2Max: 25,
  },
  spine2: {
    twistMin: -20,
    twistMax: 20,
    swing1Min: -40,
    swing1Max: 45,
    swing2Min: -30,
    swing2Max: 30,
  },
  spine3: {
    twistMin: -15,
    twistMax: 15,
    swing1Min: -45,
    swing1Max: 50,
    swing2Min: -35,
    swing2Max: 35,
  },
  neck: {
    twistMin: -30,
    twistMax: 30,
    swing1Min: -50,
    swing1Max: 60,
    swing2Min: -40,
    swing2Max: 40,
  },
  head: {
    twistMin: -45,
    twistMax: 45,
    swing1Min: -60,
    swing1Max: 70,
    swing2Min: -60,
    swing2Max: 60,
  },
  leftShoulder: {
    twistMin: -30,
    twistMax: 30,
    swing1Min: -60,
    swing1Max: 30,
    swing2Min: -30,
    swing2Max: 80,
  },
  leftUpperArm: {
    twistMin: -90,
    twistMax: 90,
    swing1Min: -160,
    swing1Max: 30,
    swing2Min: -80,
    swing2Max: 180,
  },
  leftForearm: {
    twistMin: -120,
    twistMax: 120,
    swing1Min: -30,
    swing1Max: 30,
    swing2Min: -10,
    swing2Max: 10,
  },
  leftHip: {
    twistMin: -30,
    twistMax: 30,
    swing1Min: -60,
    swing1Max: 30,
    swing2Min: -30,
    swing2Max: 60,
  },
  leftKnee: {
    twistMin: -10,
    twistMax: 10,
    swing1Min: 0,
    swing1Max: 160,
    swing2Min: -10,
    swing2Max: 10,
  },
  leftAnkle: {
    twistMin: -45,
    twistMax: 45,
    swing1Min: -45,
    swing1Max: 70,
    swing2Min: -30,
    swing2Max: 30,
  },
};

// Mirror for right side
const rightLimits: Record<string, JointLimit> = {};
for (const [key, val] of Object.entries(HUMAN_JOINT_LIMITS)) {
  if (key.startsWith("left")) {
    rightLimits["right" + key.substring(4)] = val;
  }
}
Object.assign(HUMAN_JOINT_LIMITS, rightLimits);

/**
 * Clamp a rotation angle to the joint limit.
 */
export function clampAngle(angle: number, min: number, max: number): number {
  const period = max - min;
  if (period <= 0) return min;
  let t = ((angle - min) % period + period) % period;
  return min + t;
}
