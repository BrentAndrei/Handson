/**
 * In-browser half of the render check. Drives the REAL pipeline with REAL
 * recorded clips inside a real WebGL context and screenshots each sampled
 * frame.
 *
 * Bundled on demand by scripts/render-check.ts; never imported by src/.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import { RootTranslationSolver } from "../src/avatar/rootTranslation";

declare global {
  interface Window {
    __CLIPS__?: { label: string; frames: number[][]; samples: number[] }[];
    __RUN__?: () => void;
    __DONE__?: () => void;
    __RESULT__?: any;
  }
}

function resetToBind(sk: ReturnType<typeof resolveSkeleton>, scene: THREE.Scene) {
  for (const b of sk.bones.values()) {
    b.bone.quaternion.copy(b.localRot);
    b.bone.position.copy(b.localPos);
  }
  scene.updateMatrixWorld(true);
}

async function main() {
  const clips = window.__CLIPS__ ?? [];
  if (!clips.length) throw new Error("no clips injected");

  const canvas = document.createElement("canvas");
  canvas.width = 720;
  canvas.height = 720;
  document.getElementById("stage")?.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(1);
  renderer.setSize(720, 720, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f0e8); // paper
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 1.4, 2.2);
  camera.lookAt(0, 1.0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(3, 5, 3);
  key.castShadow = true;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5);
  fill.position.set(-4, 2, -3);
  scene.add(fill);

  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync("/Avery.glb");
  gltf.scene.position.set(0, 0, 0);
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);

  const sk = resolveSkeleton(gltf.scene);
  if (!sk.getBone("hips")) throw new Error("hips bone not resolved");

  const solver = new RootTranslationSolver();
  const offset = new THREE.Vector3();

  const results: any[] = [];
  for (const clip of clips) {
    for (const i of clip.samples) {
      const raw = new Float32Array(clip.frames[i]);
      const pose = mediapipeToCanonicalPose(raw, i / clip.frames.length);
      if (!pose) continue;

      resetToBind(sk, scene);
      const result = retarget(sk, pose);
      applyBoneRotations(result.boneRotations, result.bonePositions, sk);
      scene.updateMatrixWorld(true);
      solver.solve(result.hipsWorldPos, offset);
      gltf.scene.position.copy(offset);
      gltf.scene.updateMatrixWorld(true);

      const lh = sk.getBone("leftHand")?.bone.getWorldPosition(new THREE.Vector3());
      const rh = sk.getBone("rightHand")?.bone.getWorldPosition(new THREE.Vector3());
      const head = sk.getBone("head")?.bone.getWorldPosition(new THREE.Vector3());
      const hips = sk.getBone("hips")?.bone.getWorldPosition(new THREE.Vector3());

      renderer.render(scene, camera);
      const png = renderer.domElement.toDataURL("image/png");
      const name = `clip_${clip.label}_frame${String(i).padStart(3, "0")}.png`;
      (window as any).__SCREENSHOTS__ = (window as any).__SCREENSHOTS__ || {};
      (window as any).__SCREENSHOTS__[name] = png;

      results.push({
        name,
        label: clip.label,
        frame: i,
        lh: lh?.toArray(),
        rh: rh?.toArray(),
        head: head?.toArray(),
        hips: hips?.toArray(),
        handSpan: lh && rh ? lh.distanceTo(rh) : null,
      });
    }
  }

  window.__RESULT__ = { results };
  if (window.__DONE__) window.__DONE__();
}

window.__RUN__ = main;
