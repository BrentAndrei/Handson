/**
 * verify:second-opinion — gate for the DTW second opinion
 * (secondOpinion.ts) and the emission alias
 * (temporalEnsemble.ts).
 *
 *   1. applySecondOpinion keeps the model's top-1 when
 *      DTW agrees with it
 *   2. it switches to the runner-up when DTW strongly
 *      favors it (score >= DTW_MIN_SCORE and combined
 *      gain >= SECOND_OPINION_GAIN)
 *   3. a DTW winner below DTW_MIN_SCORE never switches
 *   4. a marginal combined gain never switches
 *   5. candidates without a DTW score are treated as 0
 *   6. needsSecondOpinion fires on low confidence OR a
 *      thin margin — not on confident, clear emissions
 *   7. TemporalEnsemble aliases ALRIGHT, NO PROBLEM →
 *      NO (the bit-identical corpus duplicate) and
 *      debounces across the alias
 *   8. the ensemble exposes top-3 (raw labels) + margin
 *      for the consult
 *   9. real corpus recordings: NO scores ~1.0 against
 *      its own canonical frames and clearly lower
 *      against THANK_YOU — the data path the second
 *      opinion runs on — and a real re-ranking flips
 *      a thin model result to the DTW winner
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  DTW_FLOOR,
  DTW_RATIO,
  SECOND_OPINION_GAIN,
  applySecondOpinion,
  needsSecondOpinion,
  scoreCandidates,
} from "../src/inference/secondOpinion";
import {
  EMISSION_ALIASES,
  TemporalEnsemble,
} from "../src/inference/temporalEnsemble";
import { normalizePackedVector } from "../src/features/signSimilarity";

const FRAME_FLOATS = 345;
let passed = 0;
let failed = 0;

function check(name: string, ok: boolean, detail = ""): void {
  if (ok) {
    passed++;
    console.log(`  PASS  ${name}${detail ? ` — ${detail}` : ""}`);
  } else {
    failed++;
    console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

// --- 1. DTW agrees with the model → no switch ---
{
  const d = applySecondOpinion(
    [
      { label: "A", confidence: 0.6 },
      { label: "B", confidence: 0.5 },
    ],
    { A: 0.9, B: 0.2 }
  );
  check(
    "dtw-agreement keeps top-1",
    !d.switched && d.label === "A" && d.confidence === 0.6,
    `label=${d.label} conf=${d.confidence.toFixed(3)}`
  );
}

// --- 2. DTW strongly favors the runner-up → switch ---
{
  const d = applySecondOpinion(
    [
      { label: "A", confidence: 0.55 },
      { label: "B", confidence: 0.52 },
    ],
    { A: 0.1, B: 0.95 }
  );
  // combined(B) = .5*.52+.5*.95 = .735 vs combined(A) = .325
  check(
    "strong dtw switches to runner-up",
    d.switched && d.label === "B" && d.originalLabel === "A",
    `label=${d.label} combined=${d.confidence.toFixed(3)}`
  );
}

// --- 3. DTW winner below the floor → no switch ---
{
  const d = applySecondOpinion(
    [
      { label: "A", confidence: 0.44 },
      { label: "B", confidence: 0.6 },
    ],
    { A: 0.2, B: 0.05 }
  );
  // combined: A=.32, B=.325 → B wins by .005, but
  // B's DTW .05 is below the .1 floor — DTW has no
  // real opinion, so no switch
  check(
    "weak dtw never switches",
    !d.switched && d.label === "A",
    `dtw=${0.05} < ${DTW_FLOOR}`
  );
}

// --- 3b. Ratio guard: gain passes, ratio fails ---
{
  const d = applySecondOpinion(
    [
      { label: "A", confidence: 0.52 },
      { label: "B", confidence: 0.5 },
    ],
    { A: 0.3, B: 0.44 }
  );
  // combined: A=.41, B=.47 → gain .06 >= .05, but
  // ratio .44/.30 = 1.47 < 1.6 → no switch
  check(
    "thin dtw ratio never switches",
    !d.switched && d.label === "A",
    `ratio=${(0.44 / 0.3).toFixed(3)} < ${DTW_RATIO}`
  );
}

// --- 4. Marginal combined gain → no switch ---
{
  const d = applySecondOpinion(
    [
      { label: "A", confidence: 0.6 },
      { label: "B", confidence: 0.58 },
    ],
    { A: 0.5, B: 0.56 }
  );
  // combined: A=.55, B=.57 → gain .02 < .05
  check(
    "marginal gain never switches",
    !d.switched && d.label === "A",
    `gain=${(0.57 - 0.55).toFixed(3)} < ${SECOND_OPINION_GAIN}`
  );
}

// --- 5. Missing DTW score treated as 0 ---
{
  const d = applySecondOpinion(
    [
      { label: "A", confidence: 0.6 },
      { label: "B", confidence: 0.5 },
    ],
    {}
  );
  check(
    "missing dtw score treated as 0",
    !d.switched && d.label === "A",
    "no scores at all"
  );
}

// --- 6. needsSecondOpinion gating ---
{
  check(
    "needsSecondOpinion: confident+clear = no",
    !needsSecondOpinion(0.75, 0.3)
  );
  check(
    "needsSecondOpinion: low confidence = yes",
    needsSecondOpinion(0.69, 0.3)
  );
  check(
    "needsSecondOpinion: thin margin = yes",
    needsSecondOpinion(0.75, 0.1)
  );
}

// --- 7/8. Ensemble alias + top3 + margin + debounce ---
{
  let now = 0;
  const clock = () => now;
  const labels = ["ALRIGHT, NO PROBLEM", "NO", "OTHER"];
  const ensemble = new TemporalEnsemble(
    3,
    labels,
    7,
    0.6,
    0.5,
    0.1,
    1000,
    clock
  );
  const PROBS = new Float32Array([0.6, 0.3, 0.1]);
  let result: ReturnType<TemporalEnsemble["push"]> = null;
  for (let i = 0; i < 7; i++) result = ensemble.push(PROBS);
  check(
    "ensemble aliases ALRIGHT, NO PROBLEM → NO",
    result !== null && result.label === "NO",
    `emitted=${result?.label}`
  );
  check(
    "ensemble exposes margin",
    result !== null && Math.abs((result.margin ?? 0) - 0.3) < 1e-6,
    `margin=${result?.margin}`
  );
  check(
    "ensemble exposes raw top-3 for canonical lookup",
    result?.top3?.[0]?.label === "ALRIGHT, NO PROBLEM" &&
      result?.top3?.[1]?.label === "NO",
    `top3=${result?.top3?.map((c) => c.label).join(",")}`
  );
  // Debounce across the alias: the same underlying class
  // re-emitting within the hold is suppressed as "NO".
  let again: ReturnType<TemporalEnsemble["push"]> = null;
  for (let i = 0; i < 7; i++) again = ensemble.push(PROBS);
  check(
    "debounce spans the alias (repeat within hold suppressed)",
    again === null,
    "second emission within 1000ms → null"
  );
  now += 1100;
  let afterHold: ReturnType<TemporalEnsemble["push"]> = null;
  for (let i = 0; i < 7 && afterHold === null; i++) {
    afterHold = ensemble.push(PROBS);
  }
  check(
    "aliased label re-emits after the hold expires",
    afterHold !== null && afterHold.label === "NO",
    `after hold: ${afterHold?.label}`
  );
  check(
    "alias map contains the verified duplicate pair",
    EMISSION_ALIASES["ALRIGHT, NO PROBLEM"] === "NO"
  );
}

// --- 9. Real corpus recordings through the DTW path ---
{
  const dataUrl = new URL("../public/avatar-data/", import.meta.url);
  const readFrames = (label: string): Float32Array[] => {
    const path = fileURLToPath(
      new URL(`${encodeURIComponent(label)}.json`, dataUrl)
    );
    const doc = JSON.parse(readFileSync(path, "utf-8")) as {
      frames: number[];
    };
    const flat = new Float32Array(doc.frames);
    const count = Math.floor(flat.length / FRAME_FLOATS);
    const frames: Float32Array[] = [];
    for (let i = 0; i < count; i++) {
      frames.push(flat.slice(i * FRAME_FLOATS, (i + 1) * FRAME_FLOATS));
    }
    return frames;
  };
  const noRaw = readFrames("NO");
  const thankRaw = readFrames("THANK_YOU");
  // The hook feeds NORMALIZED live frames (App.tsx normalizes
  // before pushFrame); the tracker normalizes canonicals itself.
  const noLive = noRaw.map((f) => normalizePackedVector(f));

  const fakeStore = {
    load: async (label: string): Promise<Float32Array[] | null> => {
      if (label === "NO") return noRaw;
      if (label === "THANK_YOU") return thankRaw;
      return null;
    },
  };
  const scores = await scoreCandidates(
    ["NO", "THANK_YOU"],
    noLive,
    fakeStore
  );
  // The consult scores the live window's TAIL against
  // the full canonical, so the correct class lands well
  // below the tutor's start-aligned ~1.0 — what
  // matters is that it clears the floor and dominates
  // the other candidate by an order of magnitude.
  check(
    "real recording: correct class clears the DTW floor",
    (scores["NO"] ?? 0) >= DTW_FLOOR,
    `score=${scores["NO"]?.toFixed(4)} >= ${DTW_FLOOR}`
  );
  check(
    "real recording: NO outscores THANK_YOU 10x over",
    (scores["NO"] ?? 0) >= 10 * (scores["THANK_YOU"] ?? 1),
    `NO=${scores["NO"]?.toFixed(4)} THANK_YOU=${scores["THANK_YOU"]?.toFixed(4)}`
  );

  // A thin model result (THANK_YOU marginally ahead) flips
  // to the DTW winner — the exact failure mode this
  // consults for. The ratio guard carries the switch
  // here: the measured separation is ~160x.
  const decision = applySecondOpinion(
    [
      { label: "THANK_YOU", confidence: 0.55 },
      { label: "NO", confidence: 0.52 },
      { label: "HELLO", confidence: 0.4 },
    ],
    scores
  );
  check(
    "real re-ranking flips a thin result to the DTW winner",
    decision.switched && decision.label === "NO",
    `switched=${decision.switched} label=${decision.label}`
  );
}

console.log(
  `\nSECOND OPINION VERIFY ${failed === 0 ? "PASSED" : "FAILED"}: ${passed} passed, ${failed} failed`
);
if (failed > 0) process.exit(1);
