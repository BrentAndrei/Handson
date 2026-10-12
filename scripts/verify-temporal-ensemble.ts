/**
 * Gate for the temporal ensemble (src/inference/temporalEnsemble.ts).
 *
 * Asserts the streaming-decision contract:
 *   E1  5 consistent confident windows emit the top label
 *   E2  a partial ring never emits
 *   E3  a 3-way split has no majority and never emits
 *   E4  the same label is debounced inside the hold window
 *   E5  after the hold window the same sign can re-emit
 *   E6  ring-mean confidence below the floor never emits
 *   E7  top-1/top-2 margin below the floor never emits
 *   E8  the ring resets after an emission (fresh evidence)
 *   E9  one noisy pre-sign window can't outvote a real sign
 *  E10  a different label within the hold window still emits
 *  E11  wrong vector length throws
 */
import { TemporalEnsemble } from "../src/inference/temporalEnsemble";

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

const LABELS = ["A", "B", "C"];
const REST = 0.001;

/** Full probability vector with the given top values. */
function makeProbs(entries: Record<string, number>): Float32Array {
  const v = new Float32Array(LABELS.length).fill(REST);
  for (const [label, prob] of Object.entries(entries)) {
    v[LABELS.indexOf(label)] = prob;
  }
  return v;
}

function feed(
  ensemble: TemporalEnsemble,
  windows: Float32Array[]
): ReturnType<TemporalEnsemble["push"]> {
  let result: ReturnType<TemporalEnsemble["push"]> = null;
  for (const w of windows) result = ensemble.push(w);
  return result;
}

let t = 0;
const clock = () => t;
const strongA = () => makeProbs({ A: 0.9 });
const strongB = () => makeProbs({ B: 0.9 });

// E1: five consistent confident windows emit
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const r = feed(e, [strongA(), strongA(), strongA(), strongA(), strongA()]);
  check("E1 consistent ring emits top label",
    r !== null && r.label === "A" && Math.abs(r.confidence - 0.9) < 1e-6,
    JSON.stringify(r));
}

// E2: partial ring never emits
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const r = feed(e, [strongA(), strongA(), strongA(), strongA()]);
  check("E2 partial ring stays silent", r === null, JSON.stringify(r));
}

// E3: three-way split has no majority
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const r = feed(e, [
    makeProbs({ A: 0.8 }), makeProbs({ A: 0.8 }),
    makeProbs({ B: 0.8 }), makeProbs({ B: 0.8 }),
    makeProbs({ C: 0.8 }),
  ]);
  check("E3 split ring has no majority", r === null, JSON.stringify(r));
}

// E4: same label debounced inside the hold window
{
  t = 0;
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const first = feed(e, [strongA(), strongA(), strongA(), strongA(), strongA()]);
  t = 100;
  const again = feed(e, [strongA(), strongA(), strongA(), strongA(), strongA()]);
  check("E4 same label debounced in hold window",
    first !== null && first.label === "A" && again === null,
    `first=${JSON.stringify(first)} again=${JSON.stringify(again)}`);
}

// E5: after the hold window the sign can re-emit
{
  t = 0;
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  feed(e, [strongA(), strongA(), strongA(), strongA(), strongA()]);
  t = 1100;
  const r = feed(e, [strongA(), strongA(), strongA(), strongA(), strongA()]);
  check("E5 repeat sign emits after hold window",
    r !== null && r.label === "A", JSON.stringify(r));
}

// E6: ring-mean confidence below the floor
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const r = feed(e, [
    makeProbs({ A: 0.4 }), makeProbs({ A: 0.4 }), makeProbs({ A: 0.4 }),
    makeProbs({ A: 0.4 }), makeProbs({ A: 0.4 }),
  ]);
  check("E6 low mean confidence stays silent", r === null, JSON.stringify(r));
}

// E7: margin below the floor
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const r = feed(e, [
    makeProbs({ A: 0.55, B: 0.5 }), makeProbs({ A: 0.55, B: 0.5 }),
    makeProbs({ A: 0.55, B: 0.5 }), makeProbs({ A: 0.55, B: 0.5 }),
    makeProbs({ A: 0.55, B: 0.5 }),
  ]);
  check("E7 thin margin stays silent", r === null, JSON.stringify(r));
}

// E8: ring resets after an emission
{
  t = 0;
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  feed(e, [strongA(), strongA(), strongA(), strongA(), strongA()]);
  t = 1100;
  const r = e.push(strongA());
  check("E8 ring resets after emission", r === null, JSON.stringify(r));
}

// E9: one noisy pre-sign window can't outvote a real sign
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const noisy = makeProbs({ B: 0.06, A: 0.05 });
  const r = feed(e, [noisy, strongA(), strongA(), strongA(), strongA()]);
  check("E9 noisy window outvoted",
    r !== null && r.label === "A" && r.confidence > 0.7,
    JSON.stringify(r));
}

// E10: a different label inside the hold window still emits
{
  t = 0;
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const first = feed(e, [strongA(), strongA(), strongA(), strongA(), strongA()]);
  t = 100;
  const second = feed(e, [strongB(), strongB(), strongB(), strongB(), strongB()]);
  check("E10 label change beats debounce",
    first !== null && first.label === "A" && second !== null && second.label === "B",
    `first=${JSON.stringify(first)} second=${JSON.stringify(second)}`);
}

// E11: wrong vector length throws
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  let threw = false;
  try {
    e.push(new Float32Array([0.5, 0.5]));
  } catch {
    threw = true;
  }
  check("E11 wrong vector length throws", threw);
}

// E12: runner-up travels with the emission
{
  const e = new TemporalEnsemble(LABELS.length, LABELS, 5, 0.6, 0.5, 0.1, 1000, clock);
  const probs = () => makeProbs({ A: 0.9, B: 0.6 });
  const r = feed(e, [probs(), probs(), probs(), probs(), probs()]);
  check("E12 runner-up reported",
    r !== null &&
    r.runnerUp?.label === "B" &&
    Math.abs((r.runnerUp?.confidence ?? 0) - 0.6) < 1e-6,
    JSON.stringify(r));
}

console.log(`\n${checks} checks, ${failures} failures`);
if (failures > 0) process.exit(1);
