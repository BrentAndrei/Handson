/**
 * PHASE 10 — live <-> recorded toggle verification (browser side).
 *
 * Mounts the REAL pipeline (Xbot + resolveSkeleton + retarget + footing +
 * root solver), not a re-implementation. That distinction matters: the Phase 4
 * harness re-implements the retargeting maths, which is exactly why it missed
 * the playback gate bug found in this phase (SignAvatar gated interpolation on
 * `targetFrameRef2.current`, which the canonical loader nulls, so recorded
 * clips never animated).
 *
 * Also monkey-patches THREE.Vector3/Quaternion/Euler and counts constructions
 * while the render loop runs, so the zero-allocation claim is measured at
 * runtime rather than asserted by a text search.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton } from "../src/avatar/boneResolver";
import { solveFooting } from "../src/avatar/footingSolver";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { loadLabelDataCanonical, enableSmplxService } from "../src/avatar/dataLoader";
import { RootTranslationSolver } from "../src/avatar/rootTranslation";

const FOOTING_FLOOR_Y = -0.0029;
const SMOOTHING = 0.25;

/**
 * Allocation counting.
 *
 * ES module namespace objects are frozen, so `THREE.Vector3 = ...` throws at
 * bundle time and cannot instrument the running loop. Instead the driver
 * aliases the bare specifier "three" to `three-alloc-shim.ts`, which re-exports
 * the real module with counting subclasses. Every import of "three" in the
 * bundle -- including the one inside footingSolver.ts -- resolves to the
 * instrumented classes, so the counts cover production code paths. src/ is
 * untouched and the shim is test-only.
 */
// eslint-disable-next-line @typescript-eslint/no-var-requires
// @ts-expect-error -- test-only CommonJS shim, intentionally untyped.
import * as SHIM from "./three-alloc-shim.cjs";
const shim = SHIM as unknown as {
  __setCounting(v: boolean): void;
  __getAlloc(): { Vector3: number; Quaternion: number; Euler: number };
};
const setCounting = (v: boolean) => shim.__setCounting(v);
const getAlloc = () => shim.__getAlloc();

async function main() {
  // Surface init failures instead of hanging the driver on waitForFunction.
  window.addEventListener("error", (e) => {
    (window as unknown as Record<string, unknown>).__PHASE10_ERROR__ =
      String((e as ErrorEvent).message ?? e);
  });
  try {
    await boot();
  } catch (e) {
    (window as unknown as Record<string, unknown>).__PHASE10_ERROR__ = String(e);
  }
}

async function boot() {
  const mark = (m: string) => {
    (window as unknown as Record<string, unknown>).__PHASE10_STAGE__ = m;
  };
  mark("start");
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(640, 640, false);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 1.2, 2.6);
  camera.lookAt(0, 0.9, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
  dirLight.position.set(1, 2, 2);
  scene.add(dirLight);

  mark("pre-service");   enableSmplxService(true);
  const model = await new Promise<THREE.Group>((res, rej) => {
    new GLTFLoader().load("/Xbot.glb", (g) => res(g.scene), undefined, rej);
  });
  mark("model-loaded");   scene.add(model);
  model.updateMatrixWorld(true);
  const sk = resolveSkeleton(model);

  // Recorded playback data, loaded exactly as the component loads it.
  mark("pre-load");   const ds = await loadLabelDataCanonical("ADOPT");
  if (!ds || ds.frames.length === 0) throw new Error("ADOPT canonical data unavailable");
  const retargeted: ReturnType<typeof retarget>[] = [];
  for (const p of ds.frames) if (p) retargeted.push(retarget(sk, p));
  if (retargeted.length < 2) throw new Error("ADOPT produced too few retargeted frames");

  mark("data-loaded");   const rootSolver = new RootTranslationSolver();
  const off = new THREE.Vector3();
  const rq = new THREE.Quaternion();

  let mode: "recorded" | "live" = "recorded";
  let liveIdx = 0;
  let frameIdx = 0;
  let tgtR = retargeted[1];
  // Mirrors SignAvatar's liveSeqRef: retarget at most once per inference
  // result, NOT once per rendered frame. The live pipeline emits at ~30fps
  // while the render loop runs at 60fps, so an ungated loop would do twice the
  // production work and misreport the allocation count.
  let liveSeq = -1;

  function pushLiveFrame() {
    const fx = (window as unknown as { __PHASE10_FIXTURES__: number[][] }).__PHASE10_FIXTURES__;
    const buf = fx[liveIdx % fx.length];
    liveIdx++;
    const pose = mediapipeToCanonicalPose(Float32Array.from(buf), 0);
    if (!pose) return;
    tgtR = retarget(sk, pose);
  }

  function tick() {
    if (mode === "live") {
      if (liveSeq !== liveIdx) {
        liveSeq = liveIdx;
        pushLiveFrame();
      }
    } else {
      frameIdx = (frameIdx + 1) % retargeted.length;
      tgtR = retargeted[frameIdx];
    }
    rootSolver.solve(tgtR.hipsWorldPos, off);
    model.position.copy(off);

    for (const [name, q] of tgtR.boneRotations) {
      const bone = sk.bones.get(name)?.bone;
      if (!bone) continue;
      rq.copy(q);
      bone.quaternion.slerp(rq, SMOOTHING);
      bone.quaternion.normalize();
    }
    model.updateMatrixWorld(true);
    solveFooting(sk, { floorY: FOOTING_FLOOR_Y });
    model.updateMatrixWorld(true);
    renderer.render(scene, camera);
  }

  (window as unknown as Record<string, unknown>).__PHASE10__ = {
    start(m: "recorded" | "live") {
      applyBoneRotations(retargeted[0].boneRotations, sk);
      mode = m;
    },
    setMode(m: "recorded" | "live") {
      mode = m;
      if (m === "live") rootSolver.reset();
    },    framesRendered: 0,
    sample() {
      let nonFinite = 0;
      let badQuat = 0;
      model.updateMatrixWorld(true);
      for (const b of sk.bones.values()) {
        const e = b.bone.matrixWorld.elements;
        for (let i = 0; i < 16; i++) if (!Number.isFinite(e[i])) nonFinite++;
        const q = b.bone.quaternion;
        if (!Number.isFinite(q.x + q.y + q.z + q.w)) nonFinite++;
        const n = Math.hypot(q.x, q.y, q.z, q.w);
        if (Math.abs(n - 1) > 1e-3) badQuat++;
      }
      const pen = solveFooting(sk, { floorY: FOOTING_FLOOR_Y });
      let maxPen = 0;
      for (const r of pen) maxPen = Math.max(maxPen, r.finalPenetration);
      return { nonFinite, badQuat, maxPen, rootY: model.position.y };
    },
    allocCounts() {
      return getAlloc();
    },
    run(seconds: number) {
      setCounting(true);
      const self = (window as unknown as Record<string, unknown>).__PHASE10__ as {
        framesRendered: number;
      };
      const end = performance.now() + seconds * 1000;
      const loop = () => {
        if (performance.now() > end) {
          setCounting(false);
          return;
        }
        tick();
        self.framesRendered++;
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    },
  };

  (window as unknown as Record<string, unknown>).__PHASE10_READY__ = true;
}

void main();

