/**
 * In-browser half of the visual check. Renders each sampled clip frame in a
 * real WebGL context and returns pixel-level measurements that distinguish
 * "a signer" from "a frozen/collapsed figure":
 *
 *   - ink coverage: fraction of pixels that are not background
 *   - ink vertical extent: how tall the silhouette is
 *   - ink centroid: where the "mass" of the drawing sits
 *   - frame-to-frame pixel difference: the actual motion signal
 *
 * A frozen figure has constant coverage/centroid and ~0 motion; a signer's
 * arms move the silhouette around frame to frame.
 *
 * Bundled on demand by scripts/visual-check.ts; never imported by src/.
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

const W = 720, H = 720;

function measure(canvas: HTMLCanvasElement): any {
  const ctx = canvas.getContext("2d")!;
  const img = ctx.getImageData(0, 0, W, H);
  const d = img.data;
  // Background is the flat paper colour; anything not close to it is "ink".
  const bg = [245, 240, 232];
  let ink = 0;
  let sumX = 0, sumY = 0;
  let minY = H, maxY = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    if (Math.abs(r - bg[0]) > 18 || Math.abs(g - bg[1]) > 18 || Math.abs(b - bg[2]) > 18) {
      ink++;
      const x = (i / 4) % W;
      const y = Math.floor((i / 4) / W);
      sumX += x; sumY += y;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  const n = ink || 1;
  return {
    coverage: ink / (W * H),
    centroidX: sumX / n,
    centroidY: sumY / n,
    verticalExtent: (maxY - minY) / H,
  };
}

async function main() {
  const clips = window.__CLIPS__ ?? [];
  if (!clips.length) throw new Error("no clips injected");

  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  document.getElementById("stage")?.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(1); renderer.setSize(W, H, false);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f0e8);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 1.4, 2.2); camera.lookAt(0, 1.0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(3, 5, 3); key.castShadow = true;   scene.add(key);
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
  let prevImg: Uint8ClampedArray | null = null;
  for (const clip of clips) {
    for (const i of clip.samples) {
      const raw = new Float32Array(clip.frames[i]);
      const pose = mediapipeToCanonicalPose(raw, i / clip.frames.length);
      if (!pose) continue;

      for (const b of sk.bones.values()) {
        b.bone.quaternion.copy(b.localRot);
        b.bone.position.copy(b.localPos);
      }
      const r = retarget(sk, pose);
      applyBoneRotations(r.boneRotations, r.bonePositions, sk);
      scene.updateMatrixWorld(true);
      solver.solve(r.hipsWorldPos, offset);
      gltf.scene.position.copy(offset);
      gltf.scene.updateMatrixWorld(true);

      renderer.render(scene, camera);
      const m = measure(canvas);

      // Frame-to-frame pixel difference against the previous rendered frame.
      const ctx = canvas.getContext("2d")!;
      const cur = ctx.getImageData(0, 0, W, H).data;
      let diff = 0;
      if (prevImg) {
        for (let p = 0; p < cur.length; p += 4) {
          diff += Math.abs(cur[p] - prevImg[p]) + Math.abs(cur[p + 1] - prevImg[p + 1]) + Math.abs(cur[p + 2] - prevImg[p + 2]);
        }
        diff /= cur.length;
      }
      prevImg = cur;

      results.push({ name: `${clip.label}_f${i}`, label: clip.label, frame: i, ...m, pixelDiff: diff });
    }
  }
  window.__RESULT__ = { results };
  if (window.__DONE__) window.__DONE__();
}

window.__RUN__ = main;