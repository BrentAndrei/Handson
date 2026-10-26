/**
 * Same as render-check.client.ts but with the PHASE 44 fix DISABLED, so we
 * get a before/after comparison of the exact same frames in the exact same
 * browser. Pass __DISABLE_FIX__=true to skip the body.wrist override.
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
    __DISABLE_FIX__?: boolean;
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
  const DISABLE = window.__DISABLE_FIX__ === true;

  const canvas = document.createElement("canvas");
  canvas.width = 720; canvas.height = 720;
  document.getElementById("stage")?.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(1); renderer.setSize(720, 720, false);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f0e8);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 1.4, 2.2); camera.lookAt(0, 1.0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(3, 5, 3); key.castShadow = true; scene.add(key);
  const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5);
  fill.position.set(-4, 2, -3);
  scene.add(fill);

  const gltf = await new GLTFLoader().loadAsync("/Xbot.glb");
  gltf.scene.position.set(0, 0, 0); scene.add(gltf.scene);
  scene.updateMatrixWorld(true);

  const sk = resolveSkeleton(gltf.scene);
  if (!sk.getBone("hips")) throw new Error("hips bone not resolved");

  const solver = new RootTranslationSolver();
  const offset = new THREE.Vector3();
  const results: any[] = [];

  for (const clip of clips) {
    for (const i of clip.samples) {
      const raw = new Float32Array(clip.frames[i]);
      let pose = mediapipeToCanonicalPose(raw, i / clip.frames.length);
      if (!pose) continue;
      if (DISABLE) {
        // Revert the fix: point body.wrist back at the POSE wrist.
        pose = { ...pose, body: { ...pose.body, leftWrist: [raw[15*3+2]*-1, raw[15*3+1]*-1, raw[15*3]*-1], rightWrist: [raw[16*3+2]*-1, raw[16*3+1]*-1, raw[16*3]*-1] } };
      }

      resetToBind(sk, scene);
      const result = retarget(sk, pose);
      applyBoneRotations(result.boneRotations, result.bonePositions, sk);
      scene.updateMatrixWorld(true);
      solver.solve(result.hipsWorldPos, offset);
      gltf.scene.position.copy(offset);
      gltf.scene.updateMatrixWorld(true);

      const lh = sk.getBone("leftHand")?.bone.getWorldPosition(new THREE.Vector3());
      const rh = sk.getBone("rightHand")?.bone.getWorldPosition(new THREE.Vector3());
      renderer.render(scene, camera);
      results.push({
        name: `${DISABLE ? "old_" : "new_"}${clip.label}_f${i}`,
        label: clip.label, frame: i,
        lh: lh?.toArray(), rh: rh?.toArray(),
        handSpan: lh && rh ? lh.distanceTo(rh) : null,
      });
    }
  }
  window.__RESULT__ = { results, disabled: DISABLE };
  if (window.__DONE__) window.__DONE__();
}

window.__RUN__ = main;