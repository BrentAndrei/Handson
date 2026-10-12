/**
 * Gate for the accuracy logger (src/inference/accuracyLog.ts).
 *
 * Asserts:
 *   A1  emissions are logged with label, confidence, runner-up
 *   A2  the event cap drops the oldest events
 *   A3  events survive a storage round-trip
 *   A4  corrupt storage recovers to an empty log
 *   A5  emissionsBetween slices by time window
 *   A6  attempts carry the labels emitted during them
 *   A7  emissionStats counts labels and means confidence
 *   A8  attemptStats computes hit rate and mean score
 *   A9  clear() empties the log and persists it
 *  A10  confusionSwitches counts rapid label changes
 *  A11  a null storage degrades to memory-only without throwing
 */
import {
  AccuracyLogger,
  MAX_EVENTS,
  STORAGE_KEY,
  type AttemptEvent,
  type EmissionEvent,
} from "../src/inference/accuracyLog";

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

class FakeStorage implements Storage {
  private map = new Map<string, string>();
  get length(): number {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(key: string): string | null {
    return this.map.has(key) ? (this.map.get(key) as string) : null;
  }
  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

function emissions(log: AccuracyLogger): EmissionEvent[] {
  return log.getEvents().filter(
    (e): e is EmissionEvent => e.type === "emission"
  );
}

function attempts(log: AccuracyLogger): AttemptEvent[] {
  return log.getEvents().filter(
    (e): e is AttemptEvent => e.type === "attempt"
  );
}

let t = 0;
const clock = () => t;

// A1: emission logged with runner-up
{
  const log = new AccuracyLogger(new FakeStorage(), clock);
  log.logEmission(1000, "HELLO", 0.82, { label: "HI", confidence: 0.31 });
  const ev = emissions(log);
  check("A1 emission logged with runner-up",
    ev.length === 1 &&
    ev[0].label === "HELLO" &&
    Math.abs(ev[0].confidence - 0.82) < 1e-9 &&
    ev[0].runnerUp === "HI" &&
    Math.abs((ev[0].runnerUpConfidence ?? 0) - 0.31) < 1e-9,
    JSON.stringify(ev));
}

// A2: cap drops the oldest events
{
  const log = new AccuracyLogger(new FakeStorage(), clock);
  for (let i = 0; i < MAX_EVENTS + 10; i++) log.logEmission(i, "HELLO", 0.9);
  const ev = emissions(log);
  check("A2 event cap enforced",
    ev.length === MAX_EVENTS && ev[0].t === 10,
    `len=${ev.length} firstT=${ev[0]?.t}`);
}

// A3: storage round-trip
{
  const store = new FakeStorage();
  const log1 = new AccuracyLogger(store, clock);
  log1.logEmission(100, "HELLO", 0.9);
  log1.logEmission(200, "NO", 0.7);
  log1.flush();
  const log2 = new AccuracyLogger(store, clock);
  const ev = emissions(log2);
  check("A3 storage round-trip",
    ev.length === 2 && ev[0].label === "HELLO" && ev[1].label === "NO",
    JSON.stringify(ev));
}

// A4: corrupt storage recovers
{
  const store = new FakeStorage();
  store.setItem(STORAGE_KEY, "{not valid json");
  const log = new AccuracyLogger(store, clock);
  check("A4 corrupt storage recovers to empty",
    log.getEvents().length === 0,
    JSON.stringify(log.getEvents()));
}

// A5: emissionsBetween slices by time
{
  const log = new AccuracyLogger(new FakeStorage(), clock);
  log.logEmission(100, "A", 0.9);
  log.logEmission(200, "B", 0.9);
  log.logEmission(300, "C", 0.9);
  const mid = log.emissionsBetween(150, 250);
  check("A5 emissionsBetween slices by time",
    mid.length === 1 && mid[0].label === "B",
    JSON.stringify(mid));
}

// A6: attempts carry the labels emitted during them
{
  t = 0;
  const log = new AccuracyLogger(new FakeStorage(), clock);
  log.logEmission(100, "A", 0.9);
  log.logEmission(200, "B", 0.9);
  t = 250;
  log.logAttempt({
    target: "A",
    outcome: "scored",
    startT: 50,
    score: 0.7,
    liveFrames: 20,
    canonicalFrames: 79,
  });
  const att = attempts(log);
  check("A6 attempt carries predicted labels",
    att.length === 1 &&
    att[0].target === "A" &&
    att[0].predicted.join(",") === "A,B" &&
    Math.abs((att[0].score ?? 0) - 0.7) < 1e-9 &&
    att[0].liveFrames === 20 &&
    att[0].canonicalFrames === 79,
    JSON.stringify(att));
}

// A7: emissionStats
{
  const log = new AccuracyLogger(new FakeStorage(), clock);
  log.logEmission(1, "HELLO", 0.9);
  log.logEmission(2, "HELLO", 0.7);
  log.logEmission(3, "HELLO", 0.8);
  log.logEmission(4, "NO", 0.6);
  const stats = log.emissionStats();
  const hello = stats.find((s) => s.label === "HELLO");
  check("A7 emissionStats counts and means",
    stats.length === 2 &&
    !!hello &&
    hello.count === 3 &&
    Math.abs(hello.meanConfidence - 0.8) < 1e-9,
    JSON.stringify(stats));
}

// A8: attemptStats hit rate
{
  t = 0;
  const log = new AccuracyLogger(new FakeStorage(), clock);
  // Attempt 1: target HELLO, model emitted HELLO -> hit
  log.logEmission(100, "HELLO", 0.9);
  t = 500;
  log.logAttempt({ target: "HELLO", outcome: "scored", startT: 0, score: 0.8 });
  // Attempt 2: target HELLO, model emitted NO -> miss
  t = 600;
  log.logEmission(700, "NO", 0.9);
  t = 1000;
  log.logAttempt({ target: "HELLO", outcome: "scored", startT: 600, score: 0.4 });
  // Attempt 3: discarded
  t = 1100;
  log.logAttempt({ target: "HELLO", outcome: "discarded", startT: 1050 });
  const stats = log.attemptStats();
  const hello = stats.find((s) => s.target === "HELLO");
  check("A8 attemptStats hit rate and mean score",
    stats.length === 1 &&
    !!hello &&
    hello.attempts === 3 &&
    hello.scored === 2 &&
    hello.discarded === 1 &&
    hello.hits === 1 &&
    Math.abs(hello.hitRate - 1 / 3) < 1e-9 &&
    Math.abs((hello.meanScore ?? 0) - 0.6) < 1e-9,
    JSON.stringify(hello));
}

// A9: clear empties and persists
{
  const store = new FakeStorage();
  const log = new AccuracyLogger(store, clock);
  log.logEmission(1, "A", 0.9);
  log.clear();
  check("A9 clear empties and persists",
    log.getEvents().length === 0 &&
    store.getItem(STORAGE_KEY) === "[]",
    `events=${log.getEvents().length} stored=${store.getItem(STORAGE_KEY)}`);
}

// A10: confusion switches
{
  const log = new AccuracyLogger(new FakeStorage(), clock);
  log.logEmission(0, "A", 0.9);
  log.logEmission(500, "B", 0.9);
  log.logEmission(1500, "A", 0.9);
  log.logEmission(2000, "B", 0.9);
  const switches = log.confusionSwitches(1500);
  const ab = switches.find((s) => s.from === "A" && s.to === "B");
  const ba = switches.find((s) => s.from === "B" && s.to === "A");
  check("A10 confusion switches counted",
    !!ab && ab.count === 2 && !!ba && ba.count === 1,
    JSON.stringify(switches));
}

// A11: null storage degrades to memory-only
{
  const log = new AccuracyLogger(null, clock);
  let threw = false;
  try {
    log.logEmission(1, "A", 0.9);
    log.logAttempt({ target: "A", outcome: "scored", startT: 0 });
  } catch {
    threw = true;
  }
  check("A11 memory-only without storage",
    !threw && log.getEvents().length === 2,
    `threw=${threw} events=${log.getEvents().length}`);
}

console.log(`\n${checks} checks, ${failures} failures`);
if (failures > 0) process.exit(1);
