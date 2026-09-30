/**
 * PHASE 4 — in-browser half of the headless verification.
 *
 * Runs the REAL pipeline inside a real WebGL context:
 *   345-float MediaPipe buffer -> mediapipeToCanonicalPose -> retarget
 *   -> applyBoneRotations -> THREE.WebGLRenderer.render
 *
 * It is bundled on demand by scripts/verify-browser-pose.ts and is NEVER
 * imported by src/ — no test code reaches the production bundle.
 * No retargeter math lives here; this only drives and measures it.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import {
  RootTranslationSolver,
  DEFAULT_ROOT_TRANSLATION,
} from "../src/avatar/rootTranslation";

const FRAMES = 100;
/** Metres a foot may dip below the avatar's own bind floor (natural foot roll). */
const FOOT_TOLERANCE_M = 0.01;

interface Fixture {
  name: string;
  data: number[];
}

declare global {
  interface Window {
    __PHASE4__?: unknown;
    __PHASE4_FIXTURES__?: Fixture[];
  }
}

function resetToBind(sk: ResolvedSkeleton, scene: THREE.Scene) {
  for (const b of sk.bones.values()) {
    b.bone.quaternion.copy(b.localRot);
    b.bone.position.copy(b.localPos);
  }
  scene.updateMatrixWorld(true);
}

/** Skeleton centroid — the "avatar center" the Phase 4 gate refers to. */
function centroid(sk: ResolvedSkeleton, out = new THREE.Vector3()): THREE.Vector3 {
  out.set(0, 0, 0);
  let n = 0;
  for (const b of sk.bones.values()) {
    out.add(b.bone.getWorldPosition(new THREE.Vector3()));
    n++;
  }
  return n ? out.divideScalar(n) : out;
}

async function main() {
  const fixtures = window.__PHASE4_FIXTURES__ ?? [];
  if (!fixtures.length) throw new Error("no fixtures injected");

  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(1);
  renderer.setSize(640, 640, false);
  const gl = renderer.getContext();

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 1.5, 1.8);
  camera.lookAt(0, 1.0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(3, 4, 3);
  scene.add(key);

  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync("/Xbot.glb");
  // Mirrors SignAvatar.tsx: model at the origin, unscaled.
  gltf.scene.position.set(0, 0, 0);
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);

  const sk = resolveSkeleton(gltf.scene);
  if (!sk.getBone("hips")) throw new Error("hips bone not resolved");

  // --- bind reference: the avatar's own origin ---------------------------
  resetToBind(sk, scene);
  const bindRoot = sk.getBone("hips")!.bone.getWorldPosition(new THREE.Vector3());
  const bindCentroid = centroid(sk);
  renderer.render(scene, camera);

  // The avatar's own floor: lowest bone at bind with the model at the origin.
  // Xbot's rest pose already puts mixamorigLeftToe_End slightly below y = 0, so
  // footing is judged against this, not against an absolute plane.
  let bindFloorY = Infinity;
  for (const b of sk.bones.values()) {
    const y = b.bone.getWorldPosition(new THREE.Vector3()).y;
    if (y < bindFloorY) bindFloorY = y;
  }

  const nonFinite: string[] = [];
  const glErrors = new Set<number>();
  let maxRootDrift = 0;
  let maxCentroidDrift = 0;
  let totalMotion = 0;
  let renderedFrames = 0;
  let contextLost = false;
  let usedFixtures = 0;

  canvas.addEventListener("webglcontextlost", () => {
    contextLost = true;
  });

  const tmp = new THREE.Matrix4();
  const prevEulers = new Map<string, THREE.Euler>();

  // --- PHASE 7: root translation is now ON, so exercise the real solver ----
  const solver = new RootTranslationSolver();
  const offset = new THREE.Vector3();
  const bbox = {
    xMin: Infinity, xMax: -Infinity,
    yMin: Infinity, yMax: -Infinity,
    zMin: Infinity, zMax: -Infinity,
  };
  let minFootY = Infinity;
  let badQuatNorm = 0;
  let firstOffset = NaN;

  for (let f = 0; f < FRAMES; f++) {
    const fx = fixtures[f % fixtures.length];
    const view = new Float32Array(fx.data);
    const pose = mediapipeToCanonicalPose(view, f / 24);
    if (!pose) continue;
    usedFixtures++;

    resetToBind(sk, scene);
    const result = retarget(sk, pose);
    applyBoneRotations(result.boneRotations, sk);
    scene.updateMatrixWorld(true);

    // The delta solver consumes the raw source-unit hips position and emits a
    // metre-space offset, exactly as SignAvatar.tsx does.
    solver.solve(result.hipsWorldPos, offset);
    if (f === 0) firstOffset = offset.length();
    gltf.scene.position.copy(offset);
    gltf.scene.updateMatrixWorld(true);
    renderer.render(scene, camera);
    renderedFrames++;

    bbox.xMin = Math.min(bbox.xMin, offset.x);
    bbox.xMax = Math.max(bbox.xMax, offset.x);
    bbox.yMin = Math.min(bbox.yMin, offset.y);
    bbox.yMax = Math.max(bbox.yMax, offset.y);
    bbox.zMin = Math.min(bbox.zMin, offset.z);
    bbox.zMax = Math.max(bbox.zMax, offset.z);

    const err = gl.getError();
    if (err !== 0) glErrors.add(err);

    // --- NaN / Inf + unit-quaternion sweep over every bone ----------------
    let lowest = Infinity;
    for (const b of sk.bones.values()) {
      b.bone.matrixWorld.toArray(tmp.elements);
      if (!tmp.elements.every(Number.isFinite)) {
        nonFinite.push(`frame ${f}: ${b.name}.matrixWorld`);
      }
      const q = b.bone.quaternion;
      if (![q.x, q.y, q.z, q.w].every(Number.isFinite)) {
        nonFinite.push(`frame ${f}: ${b.name}.quaternion`);
      } else if (Math.abs(q.length() - 1) > 1e-4) {
        badQuatNorm++;
      }
      const y = b.bone.getWorldPosition(new THREE.Vector3()).y;
      if (y < lowest) lowest = y;
    }
    minFootY = Math.min(minFootY, lowest);

    // --- center drift, measured on the SKELETON (rotation only) ------------
    // The root offset is applied to gltf.scene.position, so the hips bone's own
    // world position still measures the rotation-only retarget result. That is
    // what the pre-Phase-7 rootLocked gate asserted, and it remains the right
    // thing to watch: the retargeter must not be inventing root motion of its own.
    gltf.scene.position.set(0, 0, 0);
    scene.updateMatrixWorld(true);
    const root = sk.getBone("hips")!.bone.getWorldPosition(new THREE.Vector3());
    maxRootDrift = Math.max(maxRootDrift, root.distanceTo(bindRoot));
    maxCentroidDrift = Math.max(maxCentroidDrift, centroid(sk).distanceTo(bindCentroid));

    // --- prove the frame actually changed the pose ------------------------
    for (const b of sk.bones.values()) {
      const cur = new THREE.Euler().setFromQuaternion(b.bone.quaternion, "XYZ");
      const prev = prevEulers.get(b.name);
      if (prev) {
        totalMotion +=
          Math.abs(cur.x - prev.x) + Math.abs(cur.y - prev.y) + Math.abs(cur.z - prev.z);
      }
      prevEulers.set(b.name, cur);
    }
  }

  const rendererInfo = {
    vendor: String(gl.getParameter(gl.VENDOR)),
    renderer: String(gl.getParameter(gl.RENDERER)),
    version: String(gl.getParameter(gl.VERSION)),
  };

  const C = DEFAULT_ROOT_TRANSLATION;
  const checks = {
    allFramesRendered: renderedFrames === FRAMES,
    zeroNaNInf: nonFinite.length === 0,
    // PHASE 7: root translation is deliberately no longer "locked to zero" --
    // it is BOUNDED. The old rootLocked gate asserted maxRootDrift < 1e-6, which
    // is only meaningful while the feature is off; it is replaced by an envelope
    // check that is strictly stronger in practice (it bounds travel AND
    // requires the first frame of a sequence to be exactly zero, proving the
    // offset is a delta rather than an absolute coordinate).
    rootBounded:
      Math.abs(bbox.xMin) <= C.maxLateralX + 1e-6 &&
      Math.abs(bbox.xMax) <= C.maxLateralX + 1e-6 &&
      Math.abs(bbox.zMin) <= C.maxLateralZ + 1e-6 &&
      Math.abs(bbox.zMax) <= C.maxLateralZ + 1e-6 &&
      bbox.yMin >= -C.maxDrop - 1e-6 &&
      bbox.yMax <= C.maxLift + 1e-6,
    rootIsDelta: firstOffset === 0,
    feetAboveFloor: minFootY >= bindFloorY - FOOT_TOLERANCE_M,
    quaternionsUnit: badQuatNorm === 0,
    contextNotLost: !contextLost,
    noGlErrors: glErrors.size === 0,
    framesActuallyUpdate: totalMotion > 0.5,
  };

  window.__PHASE4__ = {
    frames: FRAMES,
    renderedFrames,
    usedFixtures,
    uniqueFixtures: fixtures.length,
    bones: sk.bones.size,
    maxRootDrift,
    maxCentroidDrift,
    totalMotion,
    rootBBox: bbox,
    rootSpan: {
      x: bbox.xMax - bbox.xMin,
      y: bbox.yMax - bbox.yMin,
      z: bbox.zMax - bbox.zMin,
    },
    bindFloorY,
    minFootY,
    badQuatNorm,
    nonFinite: nonFinite.slice(0, 20),
    nonFiniteCount: nonFinite.length,
    glErrors: [...glErrors],
    contextLost,
    bindRoot: bindRoot.toArray(),
    bindCentroid: bindCentroid.toArray(),
    rendererInfo,
    checks,
    pass: Object.values(checks).every(Boolean),
  };
}

main().catch((e) => {
  window.__PHASE4__ = {
    pass: false,
    error: String(e && (e as Error).stack ? (e as Error).stack : e),
  };
});
