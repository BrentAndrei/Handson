/**
 * PHASE 11 - bit-for-bit golden snapshot of the retargeting math.
 *
 * The acceptance criterion for this phase is not "tests still pass" but "not a
 * single fraction of a degree changed". That is a much stronger claim, so this
 * captures EVERY output float of the real pipeline as a raw IEEE-754 bit
 * pattern and writes it to a JSON golden file:
 *
 *   345-float MediaPipe buffer
 *     -> mediapipeToCanonicalPose()   [captures body + both hands]
 *     -> retarget()                   [captures every bone quaternion + hips]
 *
 * Usage:
 *   npx tsx scripts/golden-snapshot.ts capture   # write the golden file
 *   npx tsx scripts/golden-snapshot.ts compare   # diff against it, exit 1 on drift
 *
 * Comparison is on raw bit patterns, not a tolerance, so a single ULP of drift
 * fails the gate.
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { createHash } from "crypto";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton } from "../src/avatar/boneResolver";
import { retarget } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import type { CanonicalPose } from "../src/avatar/canonicalPose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const GLB = path.join(ROOT, "public", "models", "Xbot.glb");
const FIXTURE_DIR = path.join(__dirname, "fixtures", "poses", "extracted");
const GOLDEN = path.join(__dirname, "fixtures", "golden-phase11.json");
const FEATURE_DIM = 345;

/** Float64 bit pattern as an exact string, so JSON round-trips losslessly. */
function bits(n: number): string {
  const b = Buffer.allocUnsafe(8);
  b.writeDoubleLE(n, 0);
  return b.toString("hex");
}

function unpack(hex: string): number {
  return Buffer.from(hex, "hex").readDoubleLE(0);
}

async function loadSkeleton() {
  const raw = fs.readFileSync(GLB);
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  const scene = new THREE.Scene();
  const gltf = await new Promise<{ scene: THREE.Group }>((res, rej) =>
    new GLTFLoader().parse(ab, "", res as never, rej)
  );
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);
  return resolveSkeleton(gltf.scene);
}

/** Capture every numeric leaf of a CanonicalPose as exact bit patterns. */
function capturePose(pose: CanonicalPose): Record<string, string> {
  const out: Record<string, string> = {};
  const vec3 = (name: string, v: readonly number[]) => {
    out[`${name}.x`] = bits(v[0]);
    out[`${name}.y`] = bits(v[1]);
    out[`${name}.z`] = bits(v[2]);
  };
  const b = pose.body;
  vec3("body.hipsPos", b.hipsPos);
  out["body.shoulderHeight"] = bits(b.shoulderHeight);
  vec3("body.spineDir", b.spineDir);
  vec3("body.shoulderAxis", b.shoulderAxis);
  vec3("body.neckPos", b.neckPos);
  vec3("body.headPos", b.headPos);
  vec3("body.headDir", b.headDir);
  for (const side of ["left", "right"] as const) {
    const S = side as never as "left" | "right";
    vec3(`body.${side}Shoulder`, b[`${S}Shoulder`]);
    vec3(`body.${side}Elbow`, b[`${S}Elbow`]);
    vec3(`body.${side}Wrist`, b[`${S}Wrist`]);
    vec3(`body.${side}Hip`, b[`${S}Hip`]);
    vec3(`body.${side}Knee`, b[`${S}Knee`]);
    vec3(`body.${side}Ankle`, b[`${S}Ankle`]);
    const h = pose[`${S}Hand`];
    vec3(`hand.${side}.wristPos`, h.wristPos);
    vec3(`hand.${side}.palmNormal`, h.palmNormal);
    vec3(`hand.${side}.palmDir`, h.palmDir);
    for (const [fname, fp] of Object.entries(h.fingers)) {
      const f = fp as { base?: number[]; tip?: number[]; joints?: Record<string, number[]> };
      if (f.base) vec3(`hand.${side}.${fname}.base`, f.base);
      if (f.tip) vec3(`hand.${side}.${fname}.tip`, f.tip);
      if (f.joints) {
        for (const [jn, jp] of Object.entries(f.joints)) vec3(`hand.${side}.${fname}.${jn}`, jp);
      }
    }
  }
  out["timestamp"] = bits(pose.timestamp);
  return out;
}

/** Capture every bone quaternion + the hips position from retarget(). */
function captureRetarget(res: ReturnType<typeof retarget>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, q] of res.boneRotations) {
    out[`rot.${name}.x`] = bits(q.x);
    out[`rot.${name}.y`] = bits(q.y);
    out[`rot.${name}.z`] = bits(q.z);
    out[`rot.${name}.w`] = bits(q.w);
  }
  for (const [name, q] of res.worldRotations) {
    out[`world.${name}.x`] = bits(q.x);
    out[`world.${name}.y`] = bits(q.y);
    out[`world.${name}.z`] = bits(q.z);
    out[`world.${name}.w`] = bits(q.w);
  }
  out["hips.x"] = bits(res.hipsWorldPos.x);
  out["hips.y"] = bits(res.hipsWorldPos.y);
  out["hips.z"] = bits(res.hipsWorldPos.z);
  return out;
}

async function build() {
  const sk = await loadSkeleton();
  const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith(".bin")).sort();
  const cases: Record<string, Record<string, string>> = {};
  let total = 0;
  for (const file of files) {
    const raw = fs.readFileSync(path.join(FIXTURE_DIR, file));
    if (raw.byteLength !== FEATURE_DIM * 4) continue;
    const view = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    const pose = mediapipeToCanonicalPose(view, 0);
    if (!pose) continue;
    const name = file.replace(/\.bin$/, "");
    cases[name] = { ...capturePose(pose), ...captureRetarget(retarget(sk, pose)) };
    total += Object.keys(cases[name]).length;
  }
  return { cases, total };
}

function hash(snapshot: Record<string, Record<string, string>>): string {
  const h = createHash("sha256");
  for (const k of Object.keys(snapshot).sort()) {
    h.update(k);
    for (const f of Object.keys(snapshot[k]).sort()) h.update(f + snapshot[k][f]);
  }
  return h.digest("hex");
}

async function main() {
  const mode = process.argv[2] ?? "capture";
  const snapshot = await build();
  const cases = Object.keys(snapshot.cases).length;
  const digest = hash(snapshot.cases);
  console.log(`PHASE 11 - BIT-FOR-BIT GOLDEN SNAPSHOT`);
  console.log(`   cases (real MediaPipe buffers) : ${cases}`);
  console.log(`   captured float values         : ${snapshot.total}`);
  console.log(`   sha256 of all bit patterns    : ${digest}`);

  if (mode === "capture") {
    fs.mkdirSync(path.dirname(GOLDEN), { recursive: true });
    fs.writeFileSync(GOLDEN, JSON.stringify(snapshot.cases));
    console.log(`\n   written -> ${path.relative(ROOT, GOLDEN)}`);
    return;
  }

  if (!fs.existsSync(GOLDEN)) {
    console.error(`\n   no golden file at ${GOLDEN} - run "capture" first`);
    process.exitCode = 1;
    return;
  }
  const golden = JSON.parse(fs.readFileSync(GOLDEN, "utf8")) as Record<string, Record<string, string>>;
  if (hash(golden) === digest) {
    console.log(`\n   GOLDEN MATCH: all ${snapshot.total} floats bit-identical.`);
    return;
  }

  let drift = 0;
  const details: string[] = [];
  for (const k of Object.keys(golden)) {
    const g = golden[k];
    const c = snapshot.cases[k];
    if (!c) { details.push(`${k}: missing in new run`); drift++; continue; }
    for (const f of Object.keys(g)) {
      if (c[f] === undefined) { details.push(`${k}.${f}: missing`); drift++; continue; }
      if (c[f] !== g[f]) {
        const a = unpack(g[f]);
        const b = unpack(c[f]);
        if (details.length < 12) details.push(`${k}.${f}: ${a} -> ${b} (d=${Math.abs(a - b).toExponential(3)})`);
        drift++;
      }
    }
  }
  console.error(`\n   GOLDEN MISMATCH: ${drift} float(s) drifted.`);
  for (const d of details) console.error(`      ${d}`);
  process.exitCode = 1;
}

void main().catch((e) => {
  console.error("GOLDEN SNAPSHOT ERROR:", e);
  process.exitCode = 1;
});

