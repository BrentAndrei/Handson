/**
 * Accuracy logger — the measurement layer for weighing
 * inference improvements.
 *
 * The repo has no labeled stream of LIVE captures: the
 * offline eval only scores the training split, so claims
 * like "the ensemble made it more accurate" are guesses
 * until events like these are collected. This module
 * records, fully on-device (localStorage — no network,
 * no video, matching the app's privacy posture):
 *
 *   emission — every label the temporal ensemble emits,
 *     with its ring-mean confidence and the runner-up
 *     (the confusion signal).
 *   attempt — every Practice attempt: the target sign,
 *     the DTW score, and which labels the model emitted
 *     while the attempt ran (the ground-truth signal:
 *     a "hit" is the target appearing in `predicted`).
 *
 * Events are capped (the oldest are dropped) and writes
 * are coalesced to 2s so a ~10/sec emission stream can't
 * jank the UI with JSON serialization. Call flush() to
 * persist immediately (tests do this).
 *
 * Inspect from the console:
 *   getAccuracyLogger().getEvents()
 *   getAccuracyLogger().emissionStats()
 *   getAccuracyLogger().attemptStats()
 *   getAccuracyLogger().confusionSwitches()
 *   getAccuracyLogger().clear()
 */

/** One model emission. */
export interface EmissionEvent {
  type: "emission";
  /** Clock timestamp (performance.now domain). */
  t: number;
  label: string;
  /** Ring-mean confidence of the emitted label. */
  confidence: number;
  runnerUp?: string;
  runnerUpConfidence?: number;
  /** Present when the DTW second opinion was consulted
   *  for this emission (secondOpinion.ts). */
  secondOpinion?: {
    originalLabel: string;
    scores: Record<string, number>;
  };
}

/** One Practice attempt against a known target sign. */
export interface AttemptEvent {
  type: "attempt";
  t: number;
  target: string;
  outcome: "scored" | "discarded";
  /** When the attempt started (slices the emissions). */
  startT: number;
  score?: number;
  liveFrames?: number;
  canonicalFrames?: number;
  /** Distinct labels the model emitted during the attempt. */
  predicted: string[];
}

export type AccuracyEvent = EmissionEvent | AttemptEvent;

function isAccuracyEvent(e: unknown): e is AccuracyEvent {
  return (
    e !== null &&
    typeof e === "object" &&
    "type" in e &&
    (((e as { type: unknown }).type === "emission" ||
      (e as { type: unknown }).type === "attempt"))
  );
}

export const STORAGE_KEY = "handson.accuracy.v1";
/** Hard cap — roughly a few hundred practice attempts
 *  plus their emission streams. Oldest events drop. */
export const MAX_EVENTS = 1000;
/** Coalesce storage writes to at most one per interval. */
const PERSIST_COALESCE_MS = 2000;

function defaultStorage(): Storage | null {
  try {
    return typeof localStorage !== "undefined" ? localStorage : null;
  } catch {
    return null;
  }
}

export class AccuracyLogger {
  private events: AccuracyEvent[] = [];
  private persistTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly storage: Storage | null = defaultStorage(),
    private readonly now: () => number = () => performance.now(),
  ) {
    this.load();
  }

  private load(): void {
    if (!this.storage) return;
    try {
      const raw = this.storage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        this.events = parsed.filter(
          (e): e is AccuracyEvent =>
            e !== null &&
            typeof e === "object" &&
            isAccuracyEvent(e)
        );
      }
    } catch {
      // Corrupt or truncated storage: start empty rather
      // than crash the app.
      this.events = [];
    }
  }

  /** Log one emission (called from the inference hook). */
  logEmission(
    t: number,
    label: string,
    confidence: number,
    runnerUp?: { label: string; confidence: number },
    secondOpinion?: { originalLabel: string; scores: Record<string, number> }
  ): void {
    this.events.push({
      type: "emission",
      t,
      label,
      confidence,
      runnerUp: runnerUp?.label,
      runnerUpConfidence: runnerUp?.confidence,
      secondOpinion,
    });
    this.trim();
    this.persistSoon();
  }

  /**
   * Log one Practice attempt. The emissions recorded
   * between startT and now are attached as `predicted`
   * (distinct labels, in order of first appearance).
   */
  logAttempt(input: {
    target: string;
    outcome: "scored" | "discarded";
    startT: number;
    score?: number;
    liveFrames?: number;
    canonicalFrames?: number;
  }): void {
    const endT = this.now();
    const predicted: string[] = [];
    const lo = this.lowerBound(input.startT);
    const hi = this.upperBound(endT);
    for (let i = lo; i < hi; i++) {
      const e = this.events[i];
      if (e.type !== "emission") continue;
      if (!predicted.includes(e.label)) predicted.push(e.label);
    }
    this.events.push({
      type: "attempt",
      t: endT,
      target: input.target,
      outcome: input.outcome,
      startT: input.startT,
      score: input.score,
      liveFrames: input.liveFrames,
      canonicalFrames: input.canonicalFrames,
      predicted,
    });
    this.trim();
    this.persistSoon();
  }

  getEvents(): AccuracyEvent[] {
    return this.events.slice();
  }

  /** Binary search for first event with t >= startT (events are time-ordered). */
  private lowerBound(startT: number): number {
    let lo = 0, hi = this.events.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.events[mid].t < startT) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** Binary search for first event with t > endT. */
  private upperBound(endT: number): number {
    let lo = 0, hi = this.events.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.events[mid].t <= endT) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** Emission events within [startT, endT] — O(log n + k) instead of O(n). */
  emissionsBetween(startT: number, endT: number): EmissionEvent[] {
    const lo = this.lowerBound(startT);
    const hi = this.upperBound(endT);
    const out: EmissionEvent[] = [];
    for (let i = lo; i < hi; i++) {
      const e = this.events[i];
      if (e.type === "emission") out.push(e);
    }
    return out;
  }

  /** Per-label emission counts and mean confidence. */
  emissionStats(): Array<{ label: string; count: number; meanConfidence: number }> {
    const byLabel = new Map<string, { count: number; sum: number }>();
    for (const e of this.events) {
      if (e.type !== "emission") continue;
      const acc = byLabel.get(e.label) ?? { count: 0, sum: 0 };
      acc.count++;
      acc.sum += e.confidence;
      byLabel.set(e.label, acc);
    }
    return Array.from(byLabel.entries())
      .map(([label, acc]) => ({
        label,
        count: acc.count,
        meanConfidence: acc.sum / acc.count,
      }))
      .sort((a, b) => b.count - a.count);
  }

  /**
   * Rapid label switches (A then B within `windowMs`) —
   * the flicker/guessing signal. A stable stream has
   * almost none of these.
   */
  confusionSwitches(windowMs = 1500): Array<{ from: string; to: string; count: number }> {
    const counts = new Map<string, number>();
    const emissions: EmissionEvent[] = [];
    for (const e of this.events) {
      if (e.type === "emission") emissions.push(e);
    }
    for (let i = 1; i < emissions.length; i++) {
      const prev = emissions[i - 1];
      const cur = emissions[i];
      if (cur.t - prev.t > windowMs || cur.label === prev.label) continue;
      const key = `${prev.label}→${cur.label}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .map(([key, count]) => {
        const [from, to] = key.split("→");
        return { from, to, count };
      })
      .sort((a, b) => b.count - a.count);
  }

  /** Per-target Practice attempt stats (hit = target was
   *  among the labels emitted during the attempt). */
  attemptStats(): Array<{
    target: string;
    attempts: number;
    scored: number;
    discarded: number;
    hits: number;
    hitRate: number;
    meanScore: number | null;
  }> {
    const byTarget = new Map<
      string,
      { attempts: number; scored: number; discarded: number; hits: number; scoreSum: number; scoreCount: number }
    >();
    for (const e of this.events) {
      if (e.type !== "attempt") continue;
      const acc = byTarget.get(e.target) ?? {
        attempts: 0, scored: 0, discarded: 0, hits: 0, scoreSum: 0, scoreCount: 0,
      };
      acc.attempts++;
      if (e.outcome === "scored") {
        acc.scored++;
        if (typeof e.score === "number") {
          acc.scoreSum += e.score;
          acc.scoreCount++;
        }
      } else {
        acc.discarded++;
      }
      if (e.predicted.includes(e.target)) acc.hits++;
      byTarget.set(e.target, acc);
    }
    return Array.from(byTarget.entries())
      .map(([target, acc]) => ({
        target,
        attempts: acc.attempts,
        scored: acc.scored,
        discarded: acc.discarded,
        hits: acc.hits,
        hitRate: acc.attempts > 0 ? acc.hits / acc.attempts : 0,
        meanScore: acc.scoreCount > 0 ? acc.scoreSum / acc.scoreCount : null,
      }))
      .sort((a, b) => b.attempts - a.attempts);
  }

  clear(): void {
    this.events = [];
    this.flush();
  }

  /** Persist immediately (also called by the 2s coalescer). */
  flush(): void {
    if (this.persistTimer !== null) {
      clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
    if (!this.storage) return;
    try {
      this.storage.setItem(STORAGE_KEY, JSON.stringify(this.events));
    } catch {
      // Quota or serialization failure: the in-memory
      // stream keeps working; this batch just isn't durable.
    }
  }

  private trim(): void {
    if (this.events.length > MAX_EVENTS) {
      this.events = this.events.slice(-MAX_EVENTS);
    }
  }

  private persistSoon(): void {
    if (this.persistTimer !== null || !this.storage) return;
    this.persistTimer = setTimeout(() => {
      this.persistTimer = null;
      this.flush();
    }, PERSIST_COALESCE_MS);
  }
}

/** App-wide singleton — the inference hook and the Practice
 *  screen both feed the same event stream. */
let singleton: AccuracyLogger | null = null;

export function getAccuracyLogger(): AccuracyLogger {
  if (!singleton) singleton = new AccuracyLogger();
  return singleton;
}
