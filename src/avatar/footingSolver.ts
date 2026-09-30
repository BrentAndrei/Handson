/**
 * PHASE 9 — additive ground-contact solver for the legs.
 *
 * ## Why this exists
 *
 * Phase 8 found that on long clips with large lower-body motion (YESTERDAY, 2240
 * frames) a foot dips below its bind floor on 16.6% of frames, worst case
 * ~2.2 cm. The root translation is NOT the cause — it is clamped to y >= 0 and
 * can only ever lift — so the penetration comes purely from hip/knee rotation.
 *
 * ## What it may and may not touch
 *
 * It may only rewrite the LOCAL quaternions of the hip (UpLeg) and knee (Leg).
 * The ankle/foot keeps its bind rotation, because Phase 4/5 established that
 * `leftAnkle`/`rightAnkle` are no-data roles that must stay at bind -- a "fix"
 * that pitched the foot would break a verified gate and would also be exactly
 * the kind of masking this phase forbids.
 *
 * ## Math
 *
 * Two-bone analytic IK per leg. With hip H, knee K, ankle A and bone lengths
 * L1 = |K-H|, L2 = |A-K|, and a target ankle T:
 *
 *   d      = |T - H|                              clamped to (|L1-L2|, L1+L2)
 *   a1     = acos((L1^2 + d^2 - L2^2) / (2 L1 d)) angle between thigh and H->T
 *   t1     = rotate(normalize(T-H), n, a1)        desired thigh direction
 *   t2     = rotate(t1, n, -(pi - interior))     desired shin direction
 *
 * where `n` is the bend-plane normal, taken from the CURRENT pose
 * (normalize(cross(u1, u2))) so the correction only changes how much the knee
 * bends, never which way it faces. If the leg is straight the cross product is
 * degenerate and a perpendicular fallback is used.
 *
 * The resulting WORLD rotations are converted to local quaternions with the
 * chain identity local = inverse(parentWorld) * world, which is the same
 * decomposition bodyRetargeter uses.
 *
 * Because the toe is a child of the ankle, raising the ankle also rotates the
 * toe, so a single pass does not land exactly on the floor. The solve therefore
 * iterates: measure the residual penetration, re-target, re-solve.
 *
 * Everything is clamped before any acos/asin, so a fully extended leg (d ->
 * L1+L2) yields a degenerate target rather than NaN.
 */
import { Quaternion, Vector3 } from "three";
import type { ResolvedSkeleton, ResolvedBone, CanonicalRole } from "./boneResolver";
import { rotationBetween } from "./canonicalPose";

export interface FootingConfig {
  /** World Y of the floor. */
  floorY: number;
  /** How far a foot may dip below floorY before a correction is applied. */
  tolerance: number;
  /** Max fixed-point passes. */
  maxIterations: number;
  /** Cap on the total per-joint correction, radians. Keeps it a *corrective*
   *  pass rather than a re-pose: it can nudge a leg, not re-aim it. */
  maxHipCorrection: number;
  maxKneeCorrection: number;
}

export const DEFAULT_FOOTING: FootingConfig = {
  floorY: 0,
  tolerance: 0.005,
  maxIterations: 3,
  maxHipCorrection: 0.35, // ~20 deg
  maxKneeCorrection: 0.6, // ~34 deg
};

export interface LegFootingResult {
  side: "left" | "right";
  /** Frames where this leg was corrected. */
  corrected: boolean;
  /** Penetration before the first pass, metres. */
  initialPenetration: number;
  /** Penetration after the final pass, metres. */
  finalPenetration: number;
  /** True if the leg was within reach of its target (otherwise clamped). */
  reachable: boolean;
  iterations: number;
}

const EPS = 1e-6;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/**
 * PHASE 10 - zero-allocation scratch pool.
 *
 * `solveFooting()` is called from inside the requestAnimationFrame loop, so
 * every `new Vector3()` on this path is ~60 short-lived objects per second per
 * leg handed straight to the GC. The solver used to create 28 of them per call
 * (2 legs x up to 3 iterations), which showed up as periodic stutter competing
 * with MediaPipe inference for the same main thread.
 *
 * Every temporary below is a module-scoped singleton reused across calls via
 * `.copy()` / `.set()`. The maths is UNCHANGED from the Phase 9 solver - only
 * the storage of the intermediates changed. `_cfg` exists so merging the
 * partial config stops allocating an object per frame too.
 *
 * Aliasing rules that must be preserved (the solver reuses these in sequence,
 * never concurrently): each role gets its own vector; helper functions take
 * their output as an explicit parameter so a caller's scratch is never
 * clobbered from the inside.
 */
const _axisX = new Vector3(1, 0, 0);
const _axisY = new Vector3(0, 1, 0);
const _axisUnit = new Vector3(1, 0, 0);
// aimWorld() scratch
const _bind = new Vector3();
const _side = new Vector3();
const _ref = new Vector3();
const _rotated = new Vector3();
const _f = new Vector3();
const _t = new Vector3();
const _cross = new Vector3();
const _dirN = new Vector3();
const _aim = new Quaternion();
const _twist = new Quaternion();
// subtree walk + length scratch
const _wpA = new Vector3();
const _wpB = new Vector3();
const _stack: ResolvedBone[] = [];
const _seen = new Set<string>();
// solveLeg scratch
const _H = new Vector3();
const _A = new Vector3();
const _T = new Vector3();
const _dir = new Vector3();
const _u1 = new Vector3();
const _u2 = new Vector3();
const _n = new Vector3();
const _t1 = new Vector3();
const _t2 = new Vector3();
const _qHipNow = new Quaternion();
const _qShinNow = new Quaternion();
const _qHipWorld = new Quaternion();
const _qShinWorld = new Quaternion();
const _qParent = new Quaternion();
const _localHip = new Quaternion();
const _localShin = new Quaternion();
const _qOutA = new Quaternion();
const _qOutB = new Quaternion();
const _cfg: FootingConfig = { ...DEFAULT_FOOTING };

// The two per-leg result objects are reused across frames. They are only ever
// read synchronously by the caller, so handing back the same two instances is
// safe -- but a caller that STORES the array would see it mutate on the next
// call. SignAvatar ignores the return value; the verify harness copies the
// numbers out immediately.
const _results: [LegFootingResult, LegFootingResult] = [
  { side: "left", corrected: false, initialPenetration: 0, finalPenetration: 0, reachable: true, iterations: 0 },
  { side: "right", corrected: false, initialPenetration: 0, finalPenetration: 0, reachable: true, iterations: 0 },
];

/** Any unit vector perpendicular to v. */
function perpendicular(v: Vector3, out: Vector3): Vector3 {
  const axis = Math.abs(v.x) < 0.9 ? _axisX : _axisY;
  return out.crossVectors(axis, v).normalize();
}

/**
 * Aiming rotation taking a bone from its rest direction to `dir`, with the
 * twist taken from the bone's CURRENT world orientation so the correction
 * cannot spin the leg about its own axis.
 */
function aimWorld(
  bone: ResolvedBone,
  dir: Vector3,
  currentWorld: Quaternion,
  out: Quaternion
): Quaternion {
  const bind = _bind.copy(bone.localDir).normalize();
  if (bind.length() < EPS) return out.identity();

  // Reference "side" of the current pose, projected perpendicular to the target.
  const side = _side.copy(_axisUnit).applyQuaternion(currentWorld);
  const ref = _ref.copy(side).projectOnPlane(dir);
  if (ref.length() < 0.1) {
    perpendicular(dir, ref);
  } else {
    ref.normalize();
  }

  rotationBetween(bind, _dirN.copy(dir).normalize(), _aim);
  const rotated = _rotated.copy(ref).applyQuaternion(_aim);

  // signed twist about the target axis, mirroring bodyRetargeter's ordering
  const f = _f.copy(ref).projectOnPlane(dir);
  const t = _t.copy(rotated).projectOnPlane(dir);
  const sign = dir.dot(_cross.crossVectors(f, t));
  const angle = Math.atan2(sign, f.dot(t));
  _twist.setFromAxisAngle(_dirN.copy(dir).normalize(), angle);
  return out.copy(_twist).multiply(_aim);
}

/**
 * Lowest world-space Y of a leg, i.e. the toe end.
 *
 * Returns a plain number rather than `{y, tip}`: the only consumer read `.y`,
 * so the tip vector and the result object were pure per-frame garbage.
 */
function lowestFootY(sk: ResolvedSkeleton, side: "left" | "right"): number {
  const ankle = sk.getBone(`${side}Ankle` as CanonicalRole);
  let lowest = Infinity;
  if (ankle) {
    // Walk the ankle's SUBTREE. The toe bones hang off the ankle and are what
    // actually touch the floor, so testing the ankle alone is not a sufficient
    // floor test. This must descend using each node's OWN children -- iterating
    // the ankle's children for every popped node never reaches Toe_End, which is
    // the lowest bone in the chain, and silently reports a clean pose.
    _stack.length = 0;
    _seen.clear();
    _stack.push(ankle);
    while (_stack.length) {
      const node = _stack.pop()!;
      if (_seen.has(node.name)) continue;
      _seen.add(node.name);
      const y = node.bone.getWorldPosition(_wpA).y;
      if (y < lowest) lowest = y;
      for (let i = 0; i < node.children.length; i++) {
        const child = sk.bones.get(node.children[i]);
        if (child) _stack.push(child);
      }
    }
  }
  return lowest;
}

/**
 * Correct one leg so its lowest foot point sits at or above floorY - tolerance.
 * Mutates the local quaternions of the hip and knee bones in place; the ankle
 * is never touched.
 */
function solveLeg(
  sk: ResolvedSkeleton,
  side: "left" | "right",
  cfg: FootingConfig
): LegFootingResult {
  const res = _results[side === "left" ? 0 : 1];
  res.side = side;
  res.corrected = false;
  res.initialPenetration = 0;
  res.finalPenetration = 0;
  res.reachable = true;
  res.iterations = 0;

  const hip = sk.getBone(`${side}Hip` as CanonicalRole);
  const knee = sk.getBone(`${side}Knee` as CanonicalRole);
  const hips = sk.getBone("hips");
  if (!hip || !knee || !hips) return res;

  const target = cfg.floorY - cfg.tolerance;
  const start = lowestFootY(sk, side);
  res.initialPenetration = Math.max(0, target - start);

  if (res.initialPenetration <= EPS) {
    res.finalPenetration = res.initialPenetration;
    return res;
  }
  res.corrected = true;

  const ankleBone = sk.getBone(`${side}Ankle` as CanonicalRole)!.bone;
  const L1 = hip.bone.getWorldPosition(_wpA)
    .distanceTo(knee.bone.getWorldPosition(_wpB));
  const L2 = knee.bone.getWorldPosition(_wpA)
    .distanceTo(ankleBone.getWorldPosition(_wpB));
  if (!(L1 > EPS) || !(L2 > EPS)) {
    res.finalPenetration = res.initialPenetration;
    return res;
  }

  for (let iter = 0; iter < cfg.maxIterations; iter++) {
    res.iterations = iter + 1;
    const pen = target - lowestFootY(sk, side);
    if (pen <= EPS) break;
    res.finalPenetration = pen;

    const H = hip.bone.getWorldPosition(_H);
    const A = ankleBone.getWorldPosition(_A);
    // Raise the ankle by the measured penetration.
    const T = _T.copy(A);
    T.y += pen;

    const dir = _dir.copy(T).sub(H);
    let d = dir.length();
    if (d < EPS) break;
    dir.divideScalar(d);
    // Clamp into the reachable annulus. This is what keeps a fully extended leg
    // (d -> L1+L2) from producing acos of an out-of-domain value.
    const dMin = Math.abs(L1 - L2) + EPS;
    const dMax = L1 + L2 - EPS;
    if (d > dMax) res.reachable = false;
    d = clamp(d, dMin, dMax);

    const cosA1 = clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1);
    const a1 = Math.acos(cosA1);
    const cosKnee = clamp((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2), -1, 1);
    const kneeInterior = Math.acos(cosKnee);

    // Bend-plane normal from the CURRENT pose, so only the bend amount changes.
    const u1 = _u1.copy(knee.bone.getWorldPosition(_wpA)).sub(H).normalize();
    const u2 = _u2.copy(A).sub(knee.bone.getWorldPosition(_wpB)).normalize();
    const n = _n.crossVectors(u1, u2);
    if (n.length() < 0.1) {
      n.crossVectors(dir, u1);
      if (n.length() < 0.1) perpendicular(dir, n);
    }
    n.normalize();

    const t1 = _t1.copy(dir).applyAxisAngle(n, a1).normalize();
    const t2 = _t2.copy(t1).applyAxisAngle(n, -(Math.PI - kneeInterior)).normalize();

    // Desired world rotations, then the chain identity local = inv(parent) * world.
    const hipWorldNow = hip.bone.getWorldQuaternion(_qHipNow);
    const shinWorldNow = knee.bone.getWorldQuaternion(_qShinNow);

    const qHipWorld = aimWorld(hip, t1, hipWorldNow, _qHipWorld);
    const qShinWorld = aimWorld(knee, t2, shinWorldNow, _qShinWorld);

    const parentWorld = hips.bone.getWorldQuaternion(_qParent);
    const localHip = _localHip.copy(parentWorld).invert().multiply(qHipWorld);
    const localShin = _localShin.copy(qHipWorld).invert().multiply(qShinWorld);

    // Bound the correction: this is a corrective pass, not a re-pose.
    hip.bone.quaternion.copy(cappedSlerp(hip.bone.quaternion, localHip, cfg.maxHipCorrection, _qOutA));
    knee.bone.quaternion.copy(cappedSlerp(knee.bone.quaternion, localShin, cfg.maxKneeCorrection, _qOutB));
    hip.bone.quaternion.normalize();
    knee.bone.quaternion.normalize();
    hip.bone.updateMatrixWorld(true);
  }

  res.finalPenetration = Math.max(0, target - lowestFootY(sk, side));
  return res;
}

/**
 * Slerp from `from` toward `to` by at most `maxAngle` radians, writing into
 * `out`. `out` must not alias `from` or `to`.
 */
function cappedSlerp(from: Quaternion, to: Quaternion, maxAngle: number, out: Quaternion): Quaternion {
  const cos = clamp(from.dot(to), -1, 1);
  const angle = 2 * Math.acos(cos);
  if (!(angle > EPS) || angle <= maxAngle) return out.copy(to);
  const t = maxAngle / angle;
  return out.copy(from).slerp(to, t);
}

/**
 * Apply ground contact to both legs. Call AFTER applyBoneRotations() and after
 * the scene matrices are up to date; returns per-leg diagnostics.
 */
export function solveFooting(
  sk: ResolvedSkeleton,
  cfg: Partial<FootingConfig> = {}
): LegFootingResult[] {
  // Merge into the module-scoped config instead of `{ ...DEFAULT, ...cfg }`,
  // which allocated a fresh object on every frame.
  const c = _cfg;
  c.floorY = cfg.floorY ?? DEFAULT_FOOTING.floorY;
  c.tolerance = cfg.tolerance ?? DEFAULT_FOOTING.tolerance;
  c.maxIterations = cfg.maxIterations ?? DEFAULT_FOOTING.maxIterations;
  c.maxHipCorrection = cfg.maxHipCorrection ?? DEFAULT_FOOTING.maxHipCorrection;
  c.maxKneeCorrection = cfg.maxKneeCorrection ?? DEFAULT_FOOTING.maxKneeCorrection;
  solveLeg(sk, "left", c);
  solveLeg(sk, "right", c);
  return _results;
}
