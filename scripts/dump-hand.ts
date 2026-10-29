/**
 * Raw hand diagnostic — dumps ACTUAL world positions of the
 * rendered finger bones alongside the MediaPipe landmarks, with
 * no auto-scaling, for a single letter recording.
 *
 * Answers one question precisely: are the finger bones landing
 * where the landmarks say the fingers are?
 *
 * Run: npx tsx scripts/dump-hand.ts A [frame]
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton, getFingerBoneNames, type ResolvedSkeleton } from "../src/avatar/boneResolver";
import { retarget, applyBoneRotations } from "../src/avatar/retargeter";
import { mediapipeToCanonicalPose, lastHandAssignment } from "../src/avatar/mediapipeAdapter";
import { gltfToCanonical, mediapipeToCanonical } from "../src/avatar/coordinateSystem";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const MODEL = path.join(ROOT, "public", "models", "Avery.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FF = 345;

const HAND_IDX = {
  WRIST: 0, THUMB_MCP: 1, THUMB_PIP: 2, THUMB_DIP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

const FINGER_ORDER = ["index", "middle", "ring", "pinky", "thumb"] as const;

async function main() {
  const rec = process.argv[2] ?? "A";
  const wantFrame = process.argv[3] ? parseInt(process.argv[3], 10) : null;

  const file = path.join(DATA_DIR, rec.endsWith(".json") ? rec : `${rec}.json`);
  const j = JSON.parse(fs.readFileSync(file, "utf8")) as { frames: number[] };
  const flat = new Float32Array(j.frames);
  const frameCount = Math.floor(flat.length / FF);

  const scene = new THREE.Scene();
  const raw = fs.readFileSync(MODEL);
  const gltf: any = await new Promise((res, rej) =>
    new GLTFLoader().parse(
      raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
      "",
      res,
      rej
    )
  );
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);
  const sk: ResolvedSkeleton = resolveSkeleton(gltf.scene);
  const names = { left: getFingerBoneNames(sk, "left"), right: getFingerBoneNames(sk, "right") };

  const reset = () => {
    for (const b of sk.bones.values()) {
      b.bone.quaternion.copy(b.localRot);
      b.bone.position.copy(b.localPos);
    }
  };
  const wc = (bone?: THREE.Object3D) => {
    if (!bone) return null;
    const p = bone.getWorldPosition(new THREE.Vector3());
    return gltfToCanonical([p.x, p.y, p.z]) as [number, number, number];
  };

  // pick frame: requested, or first with a hand
  let fi = wantFrame ?? -1;
  if (fi < 0) {
    for (let i = 0; i < frameCount; i++) {
      const f = flat.slice(i * FF, (i + 1) * FF);
      const pose = mediapipeToCanonicalPose(f, i / 24);
      if (pose && lastHandAssignment && (lastHandAssignment.left !== null || lastHandAssignment.right !== null)) {
        fi = i;
        break;
      }
    }
  }
  if (fi < 0 || fi >= frameCount) {
    console.log("no usable frame");
    return;
  }

  const frame = flat.slice(fi * FF, (fi + 1) * FF);
  const pose = mediapipeToCanonicalPose(frame, fi / 24);
  if (!pose || !lastHandAssignment) {
    console.log("pose null");
    return;
  }
  const assign = lastHandAssignment;
  console.log(`=== ${rec} frame ${fi}/${frameCount} ===`);

  reset();
  const result = retarget(sk, pose);
  applyBoneRotations(result.boneRotations, result.bonePositions, sk);
  scene.updateMatrixWorld(true);

  for (const side of ["left", "right"] as const) {
    const off = side === "left" ? assign.left : assign.right;
    if (off === null) {
      console.log(`\n-- ${side}: no hand --`);
      continue;
    }
    const lm: [number, number, number][] = [];
    for (let i = 0; i < 21; i++) {
      const b = off + i * 3;
      lm.push(mediapipeToCanonical([frame[b], frame[b + 1], frame[b + 2]]) as [number, number, number]);
    }

    console.log(`\n-- ${side} hand --`);
    console.log("RAW WORLD POSITIONS (canonical, skeleton metres — NOT rebased)");
    console.log("  landmark wrist       ", fmt(lm[HAND_IDX.WRIST]));
    console.log("  rendered wrist(hand) ", fmt(wc(sk.getBone((side === "left" ? "leftHand" : "rightHand") as any)?.bone)));

    console.log("\n  PER-FINGER: landmark MCP/PIP/DIP/TIP  vs  rendered bone world positions");
    for (const f of FINGER_ORDER) {
      const chain = names[side][f] ?? [];
      const idx = f === "thumb" ? HAND_IDX.THUMB_MCP :
        f === "index" ? HAND_IDX.INDEX_MCP :
        f === "middle" ? HAND_IDX.MIDDLE_MCP :
        f === "ring" ? HAND_IDX.RING_MCP : HAND_IDX.PINKY_MCP;
      const labels = ["MCP", "PIP", "DIP", "TIP"];
      console.log(`  ${f}:`);
      for (let k = 0; k < 4; k++) {
        const l = lm[idx + k];
        const boneName = chain[k];
        const r = boneName ? wc(sk.bones.get(boneName)?.bone) : null;
        console.log(
          `    ${labels[k]}  lm ${fmt(l)}   bone ${boneName ?? "-"} ${r ? fmt(r) : "(none)"}` +
          (r ? `   d=${dist(l, r).toFixed(3)}` : "")
        );
      }
      // localPos of the chain (what the retargeter scales)
      console.log(
        `    localPos: ` +
          chain.map((b) => (sk.bones.get(b)?.localPos.length() ?? 0).toFixed(3)).join(" / ")
      );
    }
  }
}

const fmt = (p: [number, number, number] | null) =>
  p ? `(${p[0].toFixed(3)}, ${p[1].toFixed(3)}, ${p[2].toFixed(3)})` : "(null)";
const dist = (a: [number, number, number], b: [number, number, number]) =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

main().catch((e) => { console.error(e); process.exit(1); });
