/**
 * Sign-sequence comparison and auto-advance timing for the avatar
 * playback pipeline (SignAvatar.tsx).
 *
 * ## Why this exists
 *
 * Parents pass `signSequence` as a fresh array literal on every render
 * (`glossResult?.glosses.map(g => g.gloss)` in App.tsx, `[label]` in
 * PracticeScreen.tsx). An identity check therefore sees a "new"
 * sequence on every parent re-render — every keystroke, every model
 * prediction, every 250 ms practice frame counter — and pushed that
 * into state, which tore down and rebuilt the avatar's entire WebGL
 * scene (renderer, GLB model, lights, post-processing chain). The
 * content comparison below is what gates that rebuild: the scene is
 * rebuilt only when the sequence's LABELS actually change.
 *
 * The hold-time maths serves the auto-advance through a multi-sign
 * sequence: it must wait long enough for the clip to finish playing at
 * the current playback speed, or signs get cut off mid-motion (at the
 * default 0.25x speed a 19-keyframe clip needs ~30 s, so the old
 * fixed 12 s hold discarded ~60% of every sign).
 */

/**
 * True when two sign sequences carry the same labels in the same order.
 * `undefined` and `[]` are distinct (no sequence vs. empty sequence),
 * matching how the consumers treat them.
 */
export function sameSignSequence(
  a: readonly string[] | undefined,
  b: readonly string[] | undefined,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** Per-keyframe transition duration at 1x playback speed. */
export const TRANSITION_DURATION_MS = 400;

/** Floor for the auto-advance hold: a beat of rest between signs. */
export const MIN_SIGN_HOLD_MS = 12000;

/** Extra time added past the clip's own duration before advancing. */
export const SIGN_HOLD_TAIL_MS = 1500;

/**
 * How long to hold a sign before auto-advancing to the next one.
 *
 * The clip's own duration is keyframes × per-keyframe duration ÷
 * playback speed; the hold is that plus a tail, floored at
 * MIN_SIGN_HOLD_MS so faster speeds still get a readable pause.
 * `keyframeCount <= 0` (nothing loaded yet) falls back to the floor.
 */
export function signHoldMs(keyframeCount: number, playbackSpeed: number): number {
  const speed = playbackSpeed > 1e-6 ? playbackSpeed : 1;
  const clipMs = keyframeCount > 0
    ? (keyframeCount * TRANSITION_DURATION_MS) / speed
    : 0;
  return Math.max(MIN_SIGN_HOLD_MS, clipMs + SIGN_HOLD_TAIL_MS);
}
