/**
 * Visual check computed from the rendered skeleton (no browser needed).
 *
 * The avatar's silhouette in a render is determined by its bone world
 * positions, so measuring those IS measuring the picture. For each sampled
 * frame we compute:
 *
 *   - extentY: vertical span of the whole skeleton (a signer's arms raise the
 *     top; a collapsed figure sits in a narrow band)
 *   - centroidX/Z: mean bone position in the horizontal plane (a frozen figure
 *     keeps it still; a signer's hands move it around)
 *   - handSpan: distance between the two wrists (a two-handed sign brings them
 *     together; a frozen figure keeps them fixed)
 *   - motion: frame-to-frame centroid displacement (the actual signing signal;
 *     ~0 means nothing is moving)
 *
 * A frozen/collapsed figure has ~constant extent, ~constant centroid, and
 * ~0 motion. A signer varies all three.
 *
 * Run: npx tsx scripts/visual-check.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { resolveSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import { RootTranslationSolver } from "../src/avatar/rootTranslation";
import { LandmarkSmoother } from "../src/utils/landmarkSmoother";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const GLB = path.join(ROOT, "public", "models", "Xbot.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const MEDIAPIPE_FRAME_FLOATS = 345;

function loadClip(label: string): Float32Array[] {
  const p = path.join(DATA_DIR, `${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`);
  const data = JSON.parse(fs.readFileSync(p, "utf8"));
  const frames = data.frames;
  const n = Math.floor(frames.length / MEDIAPIPE_FRAME_FLOATS);
  const out: Float32Array[] = [];
  for (let i = 0; i < n; i++) out.push(new Float32Array(frames.slice(i * MEDIAPIPE_FRAME_FLOATS, (i + 1) * MEDIAPIPE_FRAME_FLOATS)));
  return out;
}

async function loadGlb(): Promise<THREE.Group> {
  const loader = new GLTFLoader();
  const raw = fs.readFileSync(GLB);
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  return new Promise((res, rej) => loader.parse(buffer, "", (g: any) => res(g.scene), rej));
}

function measure(sk: ReturnType<typeof resolveSkeleton>): any {
  const pts: THREE.Vector3[] = [];
  for (const b of sk.bones.values()) pts.push(b.bone.getWorldPosition(new THREE.Vector3()));
  let minY = Infinity, maxY = -Infinity, cx = 0, cz = 0;
  for (const p of pts) {
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
    cx += p.x; cz += p.z;
  }
  const n = pts.length || 1;
  const lh = sk.getBone("leftHand")?.bone.getWorldPosition(new THREE.Vector3());
  const rh = sk.getBone("rightHand")?.bone.getWorldPosition(new THREE.Vector3());
  return {
    extentY: maxY - minY,
    centroidX: cx / n,
    centroidZ: cz / n,
    handSpan: lh && rh ? lh.distanceTo(rh) : null,
  };
}

async function main() {
  const scene = await loadGlb();
  const sk = resolveSkeleton(scene, { armatureScale: 0.01 });
  const solver = new RootTranslationSolver();
  const offset = new THREE.Vector3();

  const CLIPS = [
    { label: "ADOPT", samples: [0, 39, 78] },
    { label: "HELLO", samples: [0, 104, 207] },
    { label: "A", samples: [0, 47, 93] },
  ];

  let maxMotion = 0;
  let maxMotionFrame = "";
  const all: any[] = [];
  let prev: any = null;
  for (const { label, samples } of CLIPS) {
    const clip = loadClip(label);
    const smoother = new LandmarkSmoother(3, true, true);
    for (const i of samples) {
      const raw = clip[i];
      const buf = smoother.smooth(raw);
      const pose = mediapipeToCanonicalPose(buf, i / clip.length);
      if (!pose) continue;

      for (const b of sk.bones.values()) {
        b.bone.quaternion.copy(b.localRot);
        b.bone.position.copy(b.localPos);
      }
      const r = retarget(sk, pose);
      applyBoneRotations(r.boneRotations, r.bonePositions, sk);
      scene.updateMatrixWorld(true);
      solver.solve(r.hipsWorldPos, offset);
      scene.position.copy(offset);
      scene.updateMatrixWorld(true);

      const m = measure(sk);
      const motion = prev
        ? Math.abs(m.centroidX - prev.centroidX) + Math.abs(m.centroidZ - prev.centroidZ)
        : 0;
      if (motion > maxMotion) { maxMotion = motion; maxMotionFrame = `${label} f${i}`; }
      all.push({ name: `${label}_f${i}`, label, frame: i, ...m, motion });
      prev = m;
    }
  }

  console.log(JSON.stringify(all, null, 2));
  console.log(`\nmax centroid motion between consecutive frames: ${maxMotion.toFixed(3)} m at ${maxMotionFrame}`);
  const extents = all.map((a) => a.extentY);
  console.log(`vertical extent: min=${Math.min(...extents).toFixed(2)} max=${Math.max(...extents).toFixed(2)} span=${(Math.max(...extents) - Math.min(...extents)).toFixed(2)} m`);
  const spans = all.map((a) => a.handSpan).filter((s): s is number => s !== null);
  console.log(`hand span: min=${Math.min(...spans).toFixed(2)} max=${Math.max(...spans).toFixed(2)} span=${(Math.max(...spans) - Math.min(...spans)).toFixed(2)} m`);
}

main().catch((e) => { console.error(e); process.exit(1); });