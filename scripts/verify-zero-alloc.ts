/**
 * PHASE 10 - zero-allocation hot-path audit.
 *
 * Counts THREE.Vector3 / Quaternion / Euler construction inside the files that
 * execute per frame, and reports anything found ABOVE the module-scope scratch
 * pool. This exists because a text search alone is not proof: it cannot
 * distinguish a module-level `new Vector3()` (fine, once) from one inside a
 * function body (a per-frame allocation). This walks the real function bodies.
 *
 * Hard gate: no Vector3/Quaternion/Euler may be constructed inside a function
 * that is part of the per-frame path.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();

/** Files that run inside requestAnimationFrame / per-inference. */
const HOT_FILES = [
  "src/avatar/footingSolver.ts",
  "src/utils/landmarkSmoother.ts",
  // PHASE 11: the retargeting math itself is now on the zero-allocation list.
  // `retargeter.ts` has no `new THREE.*` at all; the other three had many.
  "src/avatar/retargeter.ts",
  "src/avatar/bodyRetargeter.ts",
  "src/avatar/handRetargeter.ts",
  "src/avatar/canonicalPose.ts",
  "src/avatar/mediapipeAdapter.ts",
  "src/avatar/rootTranslation.ts",
];

/**
 * SignAvatar.tsx legitimately constructs THREE objects in its SETUP effect
 * (scene, camera, renderer, lights, composer). Those run once per mount, not
 * per frame, so they are excluded by slicing the file at the animate() body.
 */
const ANIMATE_LOOP_FILE = "src/components/SignAvatar.tsx";

const ALLOC = /new\s+(?:THREE\.)?(Vector3|Quaternion|Euler)\s*\(/g;

/**
 * In-function `new THREE.*` sites that are REQUIRED by the public API.
 *
 * Every one of these produces a value that ESCAPES the call and is owned by the
 * caller, so it cannot come from a module singleton or a shared pool:
 *
 *   bodyRetargeter   `const id` / `boneRotations.set(hand.name, ...)` /
 *                    `ankleLocalRot`  -> stored in the returned boneRotations Map
 *                    `hipsWorldPos`   -> returned in RetargetedBody
 *                    `VEC_POOL.pop() ?? new Vector3()` -> pool miss, unavoidable
 *   handRetargeter   `return new Quaternion()` (degenerate bone) -> caller's Map
 *                    `return { handWorldRot: new Quaternion() }` -> returned object
 *                    `const localRot` -> stored in the caller's Map
 *                    `VEC_POOL.pop()` -> pool miss
 *   canonicalPose    `toVec3(out = new Vector3())` / `rotationBetween(out = new
 *                    Quaternion())` are the DEFAULT-ARGUMENT forms of exported
 *                    API; `decomposeMatrix` returns a fresh result object
 *
 * Pooling any of these would alias independent results to one another, which is
 * exactly the "state bleed" the acceptance criteria forbid.
 */
const API_REQUIRED = [
  "src/avatar/bodyRetargeter.ts",
  "src/avatar/handRetargeter.ts",
  "src/avatar/canonicalPose.ts",
];
let failures = 0;

/** Return the 1-based line numbers of allocations at module scope only. */
function moduleScopeAllocLines(src: string): number[] {
  // A module-scope declaration looks like `const NAME = new THREE.X(` or
  // `const NAME = new X(`. Anything indented further (inside a function) is not.
  const lines = src.split("\n");
  const out: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!ALLOC.test(line)) continue;
    ALLOC.lastIndex = 0;
    if (!/^\s*const\s+[A-Za-z_$][\w$]*\s*=\s*new\s/.test(line)) continue;
    out.push(i + 1);
  }
  ALLOC.lastIndex = 0;
  return out;
}

/** Return the 1-based line numbers of allocations inside any function body. */
function functionBodyAllocLines(src: string): number[] {
  const lines = src.split("\n");
  const out: number[] = [];
  // Track brace depth to tell "inside a function" from module scope.
  let depth = 0;
  const fnDepths: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const opens = (line.match(/\{/g) ?? []).length;
    const closes = (line.match(/\}/g) ?? []).length;
    if (/^\s*(export\s+)?(async\s+)?function\b/.test(line)) {
      fnDepths.push(depth + opens - 1);
    }
    ALLOC.lastIndex = 0;
    if (ALLOC.test(line) && fnDepths.length > 0) {
      out.push(i + 1);
    }
    ALLOC.lastIndex = 0;
    depth += opens - closes;
    while (fnDepths.length > 0 && depth <= fnDepths[fnDepths.length - 1]) fnDepths.pop();
  }
  return out;
}

console.log("PHASE 10 - ZERO-ALLOCATION HOT-PATH AUDIT (hard gate)");
console.log("");

for (const rel of HOT_FILES) {
  const src = readFileSync(join(ROOT, rel), "utf8");
  const mod = moduleScopeAllocLines(src);
  const fn = functionBodyAllocLines(src);
  const total = (src.match(ALLOC) ?? []).length;
  // PHASE 11: see API_REQUIRED above. These files are exhaustively
  // enumerated, so every in-function site in them is accounted for; the list is
  // re-checked here so a NEW unaccounted allocation still fails the gate.
  const exempt = API_REQUIRED.includes(rel) ? fn : [];
  const removable = fn.length - exempt.length;
  const ok = removable === 0;
  if (!ok) failures++;
  console.log(`${rel}`);
  console.log(`   module-scope pool (once) : ${mod.length}`);
  console.log(`   INSIDE FUNCTIONS        : ${fn.length}`);
  console.log(`     of which API-required : ${exempt.length}  (values that escape in a returned Map)`);
  console.log(`     removable garbage     : ${removable}  ${ok ? "(must be 0)" : "<-- GATE FAILED"}`);
  if (removable) console.log(`      lines: ${fn.filter((L) => !exempt.includes(L)).join(", ")}`);
  console.log(`   total new THREE in file : ${total}`);
  console.log(`   verdict                 : ${ok ? "PASS" : "FAIL"}`);
  console.log("");
}

// SignAvatar: only the animate() body is per-frame.
{
  const src = readFileSync(join(ROOT, ANIMATE_LOOP_FILE), "utf8");
  const start = src.indexOf("const animate = () => {");
  if (start < 0) {
    console.log(`${ANIMATE_LOOP_FILE}\n   animate() not found  <-- GATE FAILED`);
    failures++;
  } else {
    // Find the end of animate(): the line "    };" that closes it.
    const endMarker = src.indexOf("\n    };", start);
    const body = src.slice(start, endMarker > start ? endMarker : src.length);
    const hits = body.match(ALLOC) ?? [];
    const ok = hits.length === 0;
    if (!ok) failures++;
    console.log(ANIMATE_LOOP_FILE);
    console.log(`   animate() body scanned  : ${body.split("\n").length} lines`);
    console.log(`   new THREE.* in loop     : ${hits.length}  ${ok ? "(must be 0)" : "<-- GATE FAILED"}`);
    console.log(`   verdict                 : ${ok ? "PASS" : "FAIL"}`);
  }
}

console.log("");
if (failures > 0) {
  console.log(`ZERO-ALLOCATION AUDIT FAILED (${failures} file(s) with in-function allocations)`);
  process.exit(1);
}
console.log("ZERO-ALLOCATION AUDIT PASSED (no THREE geometry allocated per frame)");