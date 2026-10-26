/**
 * End-to-end production-path smoke test.
 *
 * The verify:* scripts test retargeter math on synthetic fixtures and the
 * browser render on extracted .bin pose files. NONE of them exercise the
 * ACTUAL production path with a real recorded clip:
 *
 *   loadLabelDataMediaPipe(label) -> mediapipeToCanonicalPose -> retarget
 *
 * A bug in that chain (a landmark-index offset, a wrong hand slot, a
 * normalization mismatch between the recorded clips and the live path) would
 * be invisible to all of them. This script wires the real pieces together
 * using the same conversion code the production path runs after fetch
 * resolves, and asserts the output is a sane pose, not a frozen/flailed avatar.
 *
 * Run: npx tsx scripts/verify-real-clip.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { resolveSkeleton } from "../src/avatar/boneResolver";
import { retarget } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import type { CanonicalPose } from "../src/avatar/canonicalPose";
import { LandmarkSmoother } from "../src/utils/landmarkSmoother";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const GLB = path.join(ROOT, "public", "models", "Xbot.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");

const MEDIAPIPE_FRAME_FLOATS = 345;
const TARGETS = ["A", "HELLO", "ADOPT", "THANK_YOU", "ARE_YOU_DEAF"];

function loadGlb(): Promise<THREE.Group> {
  const loader = new GLTFLoader();
  const raw = fs.readFileSync(GLB);
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  return new Promise((resolve, reject) => {
    loader.parse(buffer, "", (gltf: any) => resolve(gltf.scene), reject);
  });
}

/** Replicates the post-fetch half of loadLabelDataMediaPipe (dataLoader.ts:337-379). */
function loadClip(label: string): ReturnType<typeof mediapipeToCanonicalPose>[] | null {
  const sanitized = label.replace(/[^a-zA-Z0-9]/g, "_");
  const p = path.join(DATA_DIR, `${sanitized}.json`);
  if (!fs.existsSync(p)) return null;
  const data = JSON.parse(fs.readFileSync(p, "utf8"));
  if (!data || !Array.isArray(data.frames)) return null;
  const flat = new Float32Array(data.frames);
  const frameCount = Math.floor(flat.length / MEDIAPIPE_FRAME_FLOATS);
  if (frameCount === 0) return null;

  const frames: ReturnType<typeof mediapipeToCanonicalPose>[] = [];
  const smoother = new LandmarkSmoother(3, true, true);
  for (let i = 0; i < frameCount; i++) {
    const raw = flat.slice(i * MEDIAPIPE_FRAME_FLOATS, (i + 1) * MEDIAPIPE_FRAME_FLOATS);
    const buf = smoother.smooth(raw);
    frames.push(mediapipeToCanonicalPose(buf, i / frameCount));
  }
  return frames.filter((f): f is NonNullable<typeof f> => f !== null);
}

function check(label: string, scene: THREE.Group): string[] {
  const issues: string[] = [];
  const frames = loadClip(label);
  if (!frames || frames.length === 0) {
    issues.push(`${label}: clip produced no usable frames`);
    return issues;
  }
  const skel = resolveSkeleton(scene, { armatureScale: 0.01 });
  let nanCount = 0;
  let zeroBodyCount = 0;
  let handDetected = 0;
  for (const f of frames) {
    const pose = f as CanonicalPose;
    const r = retarget(skel, pose);
    for (const q of r.boneRotations.values()) {
      if (!Number.isFinite(q.x) || !Number.isFinite(q.y) || !Number.isFinite(q.z) || !Number.isFinite(q.w)) {
        nanCount++;
      }
    }
    const lh = pose.leftHand, rh = pose.rightHand;
    const lhPresent = lh && (lh.palmNormal[0] !== 0 || lh.palmNormal[1] !== 0 || lh.palmNormal[2] !== 1);
    const rhPresent = rh && (rh.palmNormal[0] !== 0 || rh.palmNormal[1] !== 0 || rh.palmNormal[2] !== 1);
    if (lhPresent || rhPresent) handDetected++;
    const la = r.boneRotations.get("leftUpperArm");
    const ra = r.boneRotations.get("rightUpperArm");
    if (la && ra && la.x === 0 && ra.x === 0) zeroBodyCount++;
  }
  issues.push(`${label}: ${frames.length} frames, NaN quats=${nanCount}, all-neutral-body=${zeroBodyCount}, hand-detected=${handDetected}`);
  if (nanCount > 0) issues.push(`  FAIL: ${nanCount} NaN quaternions`);
  if (frames.length > 5 && zeroBodyCount === frames.length) issues.push(`  FAIL: every frame has zero body rotation (frozen avatar)`);
  if (frames.length > 5 && handDetected === 0) issues.push(`  WARN: no hand detected in any frame`);
  return issues;
}

async function main() {
  const scene = await loadGlb();
  let hasSkinned = false;
  scene.traverse((o) => { if ((o as THREE.SkinnedMesh).isSkinnedMesh) hasSkinned = true; });
  if (!hasSkinned) {
    console.error("No skinned meshes in Xbot.glb");
    process.exit(1);
  }
  const all: string[] = [];
  for (const t of TARGETS) {
    all.push(...check(t, scene));
  }
  console.log(all.join("\n"));
  const fails = all.filter((l) => l.includes("FAIL"));
  console.log(fails.length === 0 ? "\nREAL-CLIP SMOKE TEST PASSED" : `\nREAL-CLIP SMOKE TEST FAILED: ${fails.length} issues`);
}
main().catch((e) => { console.error(e); process.exit(1); });