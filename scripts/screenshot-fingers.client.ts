/**
 * In-browser half of the finger screenshot harness.
 *
 * Renders the REAL pipeline (MediaPipe buffer -> mediapipeToCanonicalPose
 * -> retarget -> applyBoneRotations -> THREE.WebGLRenderer.render) on real
 * recorded signs from public/avatar-data, in a labelled grid, so a human
 * (or the driver's screenshot) can see the actual finger articulation.
 *
 * Bundled on demand by scripts/screenshot-fingers.ts; never imported by src/.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import { RootTranslationSolver } from "../src/avatar/rootTranslation";

interface CellSpec {
  label: string;
  recording: string; // "BIND" for the avatar's own rest pose
  frameFrac: number; // 0..1 position in the recording
  closeup: boolean;
}

const CELLS: CellSpec[] = [
  { label: "BIND (rest pose)", recording: "BIND", frameFrac: 0, closeup: false },
  { label: "FOUR @ 50%", recording: "FOUR", frameFrac: 0.5, closeup: false },
  { label: "FIVE @ 50%", recording: "FIVE", frameFrac: 0.5, closeup: false },
  { label: "THREE @ 50%", recording: "THREE", frameFrac: 0.5, closeup: false },
  { label: "BIND (rest pose)", recording: "BIND", frameFrac: 0, closeup: true },
  { label: "FOUR @ 50%", recording: "FOUR", frameFrac: 0.5, closeup: true },
  { label: "THANK_YOU @ 50%", recording: "THANK_YOU", frameFrac: 0.5, closeup: true },
  { label: "KNOW @ 10%", recording: "KNOW", frameFrac: 0.1, closeup: true },
];

const RECORDINGS = ["FOUR", "FIVE", "THREE", "THANK_YOU", "KNOW"];
const FRAME_FLOATS = 345;

const renderers: THREE.WebGLRenderer[] = [];

function resetToBind(sk: ResolvedSkeleton) {
  for (const b of sk.bones.values()) {
    b.bone.quaternion.copy(b.localRot);
    b.bone.position.copy(b.localPos);
  }
}

async function main() {
  const gltf = await new GLTFLoader().loadAsync("/Xbot.glb");

  const recordings: Record<string, number[]> = {};
  for (const name of RECORDINGS) {
    const res = await fetch(`/data/${name}.json`);
    const j = (await res.json()) as { frames?: number[] };
    if (!j.frames || j.frames.length < FRAME_FLOATS) {
      throw new Error(`${name}: no frames`);
    }
    recordings[name] = j.frames;
  }

  const grid = document.getElementById("grid")!;

  for (const cell of CELLS) {
    const wrap = document.createElement("div");
    wrap.className = "cell";
    const canvas = document.createElement("canvas");
    canvas.width = 360;
    canvas.height = 360;
    wrap.appendChild(canvas);
    const cap = document.createElement("div");
    cap.className = "cap";
    cap.textContent = cell.label;
    wrap.appendChild(cap);
    grid.appendChild(wrap);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(1);
    renderer.setSize(360, 360, false);
    renderers.push(renderer);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101014);
    const camera = new THREE.PerspectiveCamera(cell.closeup ? 30 : 40, 1, 0.01, 100);
    if (cell.closeup) {
      camera.position.set(0, 1.28, 0.95);
      camera.lookAt(0, 1.22, 0);
    } else {
      camera.position.set(0, 1.35, 2.6);
      camera.lookAt(0, 1.0, 0);
    }
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(2, 3, 2);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-2, 2, -1);
    scene.add(fill);

    const model = gltf.scene.clone(true);
    model.position.set(0, 0, 0);
    scene.add(model);
    scene.updateMatrixWorld(true);

    const sk = resolveSkeleton(model);
    resetToBind(sk);

    if (cell.recording !== "BIND") {
      const frames = recordings[cell.recording];
      const frameCount = Math.floor(frames.length / FRAME_FLOATS);
      const idx = Math.min(frameCount - 1, Math.max(0, Math.floor(frameCount * cell.frameFrac)));
      const view = new Float32Array(
        frames.slice(idx * FRAME_FLOATS, (idx + 1) * FRAME_FLOATS)
      );
      const pose = mediapipeToCanonicalPose(view, idx / 24);
      if (pose) {
        const result = retarget(sk, pose);
        applyBoneRotations(result.boneRotations, sk);
        const solver = new RootTranslationSolver();
        const offset = new THREE.Vector3();
        solver.solve(result.hipsWorldPos, offset);
        model.position.copy(offset);
      } else {
        cap.textContent += " (pose=null)";
      }
    }
    scene.updateMatrixWorld(true);
    renderer.render(scene, camera);
  }

  (window as any).__READY__ = true;
}

main().catch((e) => {
  (window as any).__READY__ = true;
  (window as any).__ERROR__ = String((e as Error)?.stack ?? e);
});
