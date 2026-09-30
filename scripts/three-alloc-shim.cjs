/**
 * Test-only THREE shim (PHASE 10). Re-exports the real three module but with
 * Vector3/Quaternion/Euler subclassed to count constructions, so the browser
 * driver can measure hot-path allocations at runtime instead of trusting a
 * text search. Never imported by src/.
 *
 * Why it is shaped this way:
 *  - `three/examples/jsm/*` imports "three", so it resolves to THIS file.
 *    Subclassing at module-eval time hits that circular import and the base
 *    class is still undefined ("Class extends value undefined"). The three
 *    instrumented names are therefore LAZY getters: subclass creation is
 *    deferred to first access, by which point the real module has evaluated.
 *  - Every other name is copied straight across from the real module.
 *  - CommonJS rather than ESM: in ESM, `export * from "three"` alongside
 *    same-named local exports makes the explicit ones shadow the entire star
 *    export, which breaks GLTFLoader's named imports.
 */
// Resolve to three's INTERNAL build file, never the bare "three" specifier:
// the driver aliases "three" to this file, so requiring "three" here would be
// circular and the base classes would still be undefined at subclass time.
// The absolute path is injected by the driver as __THREE_INTERNAL__.
const REAL = require(__THREE_INTERNAL__);

const ALLOC = { Vector3: 0, Quaternion: 0, Euler: 0 };
let counting = false;
const cache = {};

function make(name, key) {
  if (cache[name]) return cache[name];
  const C = class extends REAL[name] {
    constructor(...a) {
      if (counting) ALLOC[key]++;
      super(...a);
    }
  };
  Object.defineProperty(C, "name", { value: name });
  cache[name] = C;
  return C;
}

// Populate first, so the lazy getters below cannot be clobbered by the copy.
const out = {};
for (const key of Object.keys(REAL)) {
  if (key === "default") continue;
  out[key] = REAL[key];
}
for (const name of ["Vector3", "Quaternion", "Euler"]) {
  Object.defineProperty(out, name, {
    configurable: true,
    enumerable: true,
    get: () => make(name, name),
  });
}
out.__setCounting = (v) => { counting = v; };
out.__getAlloc = () => ({ Vector3: ALLOC.Vector3, Quaternion: ALLOC.Quaternion, Euler: ALLOC.Euler });

module.exports = out;
