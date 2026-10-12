/**
 * Gate for the avatar sign-sequence helpers (src/avatar/signSequence.ts).
 *
 * Asserts:
 *   S1  same array identity compares equal
 *   S2  same labels in a FRESH array compare equal (the flicker fix:
 *       parents pass a new array literal every render)
 *   S3  different lengths compare unequal
 *   S4  a single different label compares unequal
 *   S5  different order compares unequal
 *   S6  undefined vs array compares unequal
 *   S7  undefined vs undefined compares equal
 *   S8  empty vs empty compares equal
 *   S9  hold at 0.25x covers the full 19-keyframe clip (~30.4 s)
 *  S10  hold at 1x/2x floors at MIN_SIGN_HOLD_MS (12 s)
 *  S11  hold for an unloaded clip (0 keyframes) floors at 12 s
 *  S12  hold shrinks as playback speed rises (until the floor)
 *  S13  hold is finite and positive for a zero/negative speed
 */
import {
  sameSignSequence,
  signHoldMs,
  MIN_SIGN_HOLD_MS,
  TRANSITION_DURATION_MS,
  SIGN_HOLD_TAIL_MS,
} from "../src/avatar/signSequence";

let checks = 0;
let failures = 0;

function check(name: string, ok: boolean, detail = ""): void {
  checks++;
  if (!ok) {
    failures++;
    console.log(`  FAIL ${name}${detail ? `: ${detail}` : ""}`);
  } else {
    console.log(`  ok   ${name}`);
  }
}

const SEQ = ["HELLO", "THANK YOU", "HOW ARE YOU"];

// S1: identity
check("S1 same identity compares equal", sameSignSequence(SEQ, SEQ) === true);

// S2: content equality across identity (THE fix)
{
  const a = ["HELLO", "THANK YOU"];
  const b = ["HELLO", "THANK YOU"];
  check("S2 fresh array with same labels compares equal",
    sameSignSequence(a, b) === true,
    "identity-only comparison would return false here");
}

// S3: length mismatch
check("S3 different lengths compare unequal",
  sameSignSequence(["HELLO"], ["HELLO", "THANK YOU"]) === false);

// S4: one label differs
check("S4 single differing label compares unequal",
  sameSignSequence(["HELLO", "THANK YOU"], ["HELLO", "PLEASE"]) === false);

// S5: order matters
check("S5 reordered sequence compares unequal",
  sameSignSequence(["HELLO", "THANK YOU"], ["THANK YOU", "HELLO"]) === false);

// S6: undefined vs array
check("S6 undefined vs array compares unequal",
  sameSignSequence(undefined, ["HELLO"]) === false &&
    sameSignSequence(["HELLO"], undefined) === false);

// S7: undefined vs undefined
check("S7 undefined vs undefined compares equal",
  sameSignSequence(undefined, undefined) === true);

// S8: empty vs empty
check("S8 empty vs empty compares equal",
  sameSignSequence([], []) === true);

// S9: 0.25x hold covers a 19-keyframe clip
{
  const keyframes = 19;
  const clipMs = (keyframes * TRANSITION_DURATION_MS) / 0.25;
  const hold = signHoldMs(keyframes, 0.25);
  check("S9 0.25x hold covers full clip duration",
    hold >= clipMs && Math.abs(hold - (clipMs + SIGN_HOLD_TAIL_MS)) < 1e-6,
    `hold=${hold} clip=${clipMs}`);
}

// S10: 1x and 2x floor at MIN_SIGN_HOLD_MS
{
  const oneX = signHoldMs(19, 1);
  const twoX = signHoldMs(19, 2);
  check("S10 1x/2x holds floor at MIN_SIGN_HOLD_MS",
    oneX === MIN_SIGN_HOLD_MS && twoX === MIN_SIGN_HOLD_MS,
    `1x=${oneX} 2x=${twoX}`);
}

// S11: nothing loaded yet
check("S11 zero keyframes floors at MIN_SIGN_HOLD_MS",
  signHoldMs(0, 1) === MIN_SIGN_HOLD_MS);

// S12: hold decreases as speed rises until the floor
{
  const quarter = signHoldMs(19, 0.25);
  const half = signHoldMs(19, 0.5);
  check("S12 hold shrinks with faster playback (until floor)",
    quarter > half && half >= MIN_SIGN_HOLD_MS,
    `0.25x=${quarter} 0.5x=${half}`);
}

// S13: degenerate speed stays finite/positive
{
  const zero = signHoldMs(19, 0);
  const neg = signHoldMs(19, -3);
  check("S13 zero/negative speed yields finite positive hold",
    Number.isFinite(zero) && zero > 0 && Number.isFinite(neg) && neg > 0,
    `zero=${zero} neg=${neg}`);
}

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  process.exit(1);
}
