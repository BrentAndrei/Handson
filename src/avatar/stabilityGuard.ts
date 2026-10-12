/**
 * PHASE 44 — temporal stability lock.
 *
 * The retarget is deterministic and faithful, but two upstream
 * defects make individual frames unreliable:
 *
 *   1. HAND DROPOUT. MediaPipe loses a hand for stretches of
 *      frames (measured across the shipped clips: a hand is
 *      absent 19-100% of frames depending on the sign). The
 *      adapter fills an absent hand with a fixed placeholder
 *      (palmNormal/palmDir = (0,0,1), five straight fingers).
 *      That placeholder is NOT a measurement, so it must not
 *      participate in flip detection: comparing it against the
 *      last real palm normal flags every dropout — and every
 *      recovery — as a flip, which froze the avatar for over
 *      half of every clip (measured held rate 56.4%).
 *
 *   2. HANDEDNESS ASSIGNMENT. resolveHandAssignment is a
 *      nearest-pose-wrist match that flickers on near-ties,
 *      inverting the signed palm normal (~180 deg) for a frame
 *      or two — the visible "flailing".
 *
 * This module addresses both:
 *
 *   - isHandFlip ignores placeholder hands on BOTH sides of the
 *     comparison, so a dropout or a recovery is never mistaken
 *     for a flip. Only hands present (non-placeholder) in BOTH
 *     frames are compared.
 *
 *   - A frame whose palm normal genuinely inverts (> 90 deg vs
 *     the last good frame — impossible hand speed, so an error)
 *     is replaced by the last known-good pose, so the avatar
 *     locks into the correct pose through a handedness failure.
 *
 *   - A placeholder hand is re-dressed with the last good
 *     hand's orientation (palmNormal/palmDir/finger curls) so
 *     the hand holds its last shape through a dropout instead
 *     of snapping to a stiff flat pose. The wrist position is
 *     kept fresh from the body tracker, which still sees the
 *     wrist even when the hand detector does not.
 *
 * Detection of a genuine flip: a handedness flip inverts the
 * palm normal, so
 *   dot(palmNormal_now, palmNormal_lastGood) < 0   (> 90 deg)
 * Normal signing motion changes the palm normal by only ~15-25 deg
 * per frame (measured on the shipped clips), so a 90 deg threshold
 * sits in a wide, empty band between "fast but real" motion and
 * "flipped".
 */
import type { CanonicalPose, HandPose, FingerName, Vec3 } from "./canonicalPose";
import type { ResolvedSkeleton } from "./boneResolver";
import { retarget, type RetargetedPose } from "./retargeter";

/** Palm-normal dot below this = a handedness flip (> 90 deg inversion). */
const FLIP_DOT_THRESHOLD = 0.0;

/**
 * Maximum consecutive frames the lock will hold the last good
 * pose before it RE-SYNCS to the current pose.
 *
 * A flip is a >90 deg palm-normal change in a single frame —
 * faster than any human hand can move — so the flipped frame is
 * always a data error, and holding the last good pose through it
 * is correct. But the error can PERSIST: the tracker can lock
 * onto a wrong orientation for a long stretch (measured: 58 and
 * 143 frame stretches in the shipped clips). Holding those
 * freezes the avatar for seconds.
 *
 * So the hold is bounded. A transient flip (1-3 frames, the
 * common case — measured hold runs are overwhelmingly length
 * 1-3) is still held and invisible. A persistent disagreement
 * re-syncs after MAX_HOLD frames: the current pose is accepted
 * as the new reference and the avatar resumes animating. The
 * post-flip orientation is stable in every measured persistent
 * stretch, so the re-synced reference does not itself flicker.
 *
 * 5 frames ≈ 170 ms at 30 fps — long enough to ride out a
 * transient error, short enough that no freeze is perceptible.
 */
export const MAX_HOLD = 5;

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** The five finger names in canonical order. */
const FINGER_NAMES: FingerName[] = ["thumb", "index", "middle", "ring", "pinky"];

/**
 * True when `hand` is the adapter's placeholder for an absent hand.
 *
 * The placeholder carries a fixed palmNormal/palmDir of (0,0,1) and
 * five fingers all pointing (0,0,1) with exactly zero curl. A real
 * hand cannot produce that signature — five normalized segment
 * directions all exactly (0,0,1) — so it is a reliable marker for
 * "no hand was detected this frame".
 */
export function isPlaceholderHand(hand: HandPose): boolean {
  if (
    hand.palmNormal[0] !== 0 ||
    hand.palmNormal[1] !== 0 ||
    hand.palmNormal[2] !== 1
  ) {
    return false;
  }
  if (hand.palmDir[0] !== 0 || hand.palmDir[1] !== 0 || hand.palmDir[2] !== 1) {
    return false;
  }
  for (const fn of FINGER_NAMES) {
    const f = hand.fingers[fn];
    if (!f) continue;
    if (f.curl !== 0) return false;
    if (f.dir[0] !== 0 || f.dir[1] !== 0 || f.dir[2] !== 1) return false;
  }
  return true;
}

/**
 * True when either hand's palm normal inverted (> 90 deg) relative
 * to the reference pose — the signature of a handedness-assignment
 * swap.
 *
 * Placeholder (absent) hands are skipped on BOTH sides: an absent
 * hand carries no orientation, so it must never trigger or block a
 * flip. Otherwise every dropout and every recovery would register
 * as a flip and the lock would freeze the avatar.
 */
export function isHandFlip(pose: CanonicalPose, ref: CanonicalPose): boolean {
  if (
    !isPlaceholderHand(pose.leftHand) &&
    !isPlaceholderHand(ref.leftHand) &&
    dot3(pose.leftHand.palmNormal, ref.leftHand.palmNormal) < FLIP_DOT_THRESHOLD
  ) {
    return true;
  }
  if (
    !isPlaceholderHand(pose.rightHand) &&
    !isPlaceholderHand(ref.rightHand) &&
    dot3(pose.rightHand.palmNormal, ref.rightHand.palmNormal) < FLIP_DOT_THRESHOLD
  ) {
    return true;
  }
  return false;
}

/**
 * Re-dress a placeholder hand with the last good hand's orientation.
 *
 * The fresh hand's wrist position is kept — the body tracker still
 * reports the wrist when the hand detector does not — while the
 * orientation (palm normal, palm direction and the five finger
 * curls/directions) is held from the last good frame, so the hand
 * keeps its last shape through a dropout instead of snapping to a
 * stiff flat pose. Returns `fresh` unchanged when there is nothing
 * worth holding.
 */
function holdHandThroughDropout(fresh: HandPose, held: HandPose): HandPose {
  if (isPlaceholderHand(held)) return fresh;
  return {
    ...fresh,
    palmNormal: held.palmNormal,
    palmDir: held.palmDir,
    fingers: held.fingers,
  };
}

/**
 * Substitute held orientations for any placeholder hand in `pose`.
 * Returns `pose` itself (no allocation) when both hands are real.
 */
function holdHandsThroughDropout(
  pose: CanonicalPose,
  lastGood: CanonicalPose | null
): CanonicalPose {
  if (lastGood === null) return pose;
  const leftPh = isPlaceholderHand(pose.leftHand);
  const rightPh = isPlaceholderHand(pose.rightHand);
  if (!leftPh && !rightPh) return pose;
  return {
    ...pose,
    leftHand: leftPh
      ? holdHandThroughDropout(pose.leftHand, lastGood.leftHand)
      : pose.leftHand,
    rightHand: rightPh
      ? holdHandThroughDropout(pose.rightHand, lastGood.rightHand)
      : pose.rightHand,
  };
}

/**
 * Stabilize a full recorded frame sequence (the playback path).
 *
 * Runs BEFORE the 16-frame sampling, so the lock uses per-frame
 * continuity (the correct ~15-25 deg/frame baseline) rather than
 * the sparse sampling interval, where legitimate motion between
 * keyframes could exceed 90 deg and false-trigger. A frame whose
 * palm normal inverted vs the last GOOD frame is replaced by that
 * last good pose (held) for at most MAX_HOLD frames, after which
 * the lock re-syncs to the current pose so a persistent error
 * cannot freeze the avatar. A placeholder hand is re-dressed with
 * the last good orientation.
 *
 * Cheap: it only reads palm normals (dot products); it never
 * retargets.
 */
export function stabilizeCanonicalFrames(
  frames: (CanonicalPose | null)[]
): (CanonicalPose | null)[] {
  const out: (CanonicalPose | null)[] = new Array(frames.length);
  let lastGood: CanonicalPose | null = null;
  let holdCount = 0;
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (f === null) {
      out[i] = f;
      continue;
    }
    if (lastGood !== null && isHandFlip(f, lastGood)) {
      holdCount++;
      if (holdCount > MAX_HOLD) {
        // RE-SYNC: the disagreement has persisted longer than a
        // transient error can, so accept the current pose as the
        // new reference and resume animating.
        const resynced = holdHandsThroughDropout(f, lastGood);
        out[i] = resynced;
        lastGood = resynced;
        holdCount = 0;
      } else {
        // LOCK: hold the last good pose through the flipped frame.
        out[i] = lastGood;
      }
    } else {
      holdCount = 0;
      // Hold the last good hand shape through any dropout, then
      // advance the reference so the next frame compares against
      // a pose whose hands are all real measurements.
      const stabilized = holdHandsThroughDropout(f, lastGood);
      out[i] = stabilized;
      lastGood = stabilized;
    }
  }
  return out;
}

/**
 * Live stability lock (the real-time path).
 *
 * Wraps retarget(): a frame that flips the palm normal returns
 * the last known-good RetargetedPose unchanged for at most
 * MAX_HOLD frames, then re-syncs to the current pose so a
 * persistent error cannot freeze the avatar. A placeholder hand
 * is re-dressed with the last good orientation. The caller
 * (SignAvatar's render loop) reads bone rotations out of the
 * returned pose via slerpQuaternions and never mutates it, so
 * returning the held reference is safe and allocation-free on
 * the hold path.
 */
export class StabilityGuard {
  private lastGoodPose: CanonicalPose | null = null;
  private lastGood: RetargetedPose | null = null;
  private holdCount = 0;
  private flipsBlocked = 0;
  private framesSeen = 0;

  /** Retarget `pose`, holding the last good pose across a handedness flip. */
  retargetStable(
    skeleton: ResolvedSkeleton,
    pose: CanonicalPose
  ): RetargetedPose {
    this.framesSeen++;
    if (
      this.lastGoodPose !== null &&
      this.lastGood !== null &&
      isHandFlip(pose, this.lastGoodPose)
    ) {
      this.holdCount++;
      if (this.holdCount > MAX_HOLD) {
        // RE-SYNC: persistent disagreement — accept this pose as
        // the new reference and resume animating.
        const stabilized = holdHandsThroughDropout(pose, this.lastGoodPose);
        const r = retarget(skeleton, stabilized);
        this.lastGoodPose = stabilized;
        this.lastGood = r;
        this.holdCount = 0;
        return r;
      }
      this.flipsBlocked++;
      return this.lastGood; // LOCK — hold the last good pose
    }
    this.holdCount = 0;
    const stabilized = holdHandsThroughDropout(pose, this.lastGoodPose);
    const r = retarget(skeleton, stabilized);
    this.lastGoodPose = stabilized;
    this.lastGood = r;
    return r;
  }

  /** Forget the held pose (new sequence / source switch / camera toggle). */
  reset(): void {
    this.lastGoodPose = null;
    this.lastGood = null;
    this.holdCount = 0;
    this.flipsBlocked = 0;
    this.framesSeen = 0;
  }

  /** How many flips were blocked out of how many frames seen (telemetry). */
  get stats(): { flipsBlocked: number; framesSeen: number } {
    return { flipsBlocked: this.flipsBlocked, framesSeen: this.framesSeen };
  }
}
