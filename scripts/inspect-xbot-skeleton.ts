/**
 * Debug script: inspect the Xbot GLB skeleton.
 * Prints complete bone hierarchy with local/world transforms.
 * Run: npx tsx scripts/inspect-xbot-skeleton.ts
 */
import path from "path";
import { fileURLToPath } from "url";
import * as fs from "fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton } from "../src/avatar/boneResolver";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MODEL_PATH = path.join(__dirname, "..", "public", "models", "Xbot.glb");

async function main() {
  const loader = new GLTFLoader();
  const buffer = fs.readFileSync(MODEL_PATH);
  const gltf = await new Promise<any>((resolve, reject) => {
    loader.parse(buffer, "", resolve, undefined, (err) => reject(err));
  });

  const scene = gltf.scene;
  const resolved = resolveSkeleton(scene);

  console.log("\n========== XVOT SKELETON REPORT ==========\n");
  console.log(`Total bones in resolved skeleton: ${resolved.boneOrder.length}`);
  console.log(`Bone order: ${resolved.boneOrder.join(", ")}\n`);

  // For each bone in resolved map, print full info
  console.log("--- RESOLVED BONES ---");
  for (const [name, entry] of resolved.bones.entries()) {
    const bone = entry.bone;
    const parentBone = bone.parent as THREE.Bone;
    const parentName = parentBone?.userData?.name || parentBone?.name || "(root)";

    // Get children
    const childNames: string[] = [];
    for (const child of bone.children) {
      if (child instanceof THREE.Bone) {
        childNames.push(child.name || "(unnamed)");
      }
    }

    // World transforms
    bone.updateWorldMatrix(true, true);
    const worldPos = new THREE.Vector3();
    const worldQuat = new THREE.Quaternion();
    bone.getWorldPosition(worldPos);
    bone.getWorldQuaternion(worldQuat);

    // Direction to first child
    let childDir = new THREE.Vector3();
    if (childNames.length > 0) {
      const firstChild = bone.children.find(c => c instanceof THREE.Bone) as THREE.Bone;
      if (firstChild) {
        const childWorldPos = new THREE.Vector3();
        firstChild.getWorldPosition(childWorldPos);
        childDir.subVectors(childWorldPos, worldPos).normalize();
      }
    }

    // Bone length
    const boneLength = bone.position.length();

    console.log(`\nBone: ${name}`);
    console.log(`  Parent: ${parentName}`);
    console.log(`  Children: [${childNames.join(", ")}]`);
    console.log(`  Local Pos:  [${bone.position.x.toFixed(4)}, ${bone.position.y.toFixed(4)}, ${bone.position.z.toFixed(4)}]`);
    console.log(`  Local Quat: [${bone.quaternion.x.toFixed(4)}, ${bone.quaternion.y.toFixed(4)}, ${bone.quaternion.z.toFixed(4)}, ${bone.quaternion.w.toFixed(4)}]`);
    console.log(`  Local Scale: [${bone.scale.x.toFixed(4)}, ${bone.scale.y.toFixed(4)}, ${bone.scale.z.toFixed(4)}]`);
    console.log(`  World Pos:  [${worldPos.x.toFixed(4)}, ${worldPos.y.toFixed(4)}, ${worldPos.z.toFixed(4)}]`);
    console.log(`  World Quat: [${worldQuat.x.toFixed(4)}, ${worldQuat.y.toFixed(4)}, ${worldQuat.z.toFixed(4)}, ${worldQuat.w.toFixed(4)}]`);
    console.log(`  Bone length (pos mag): ${boneLength.toFixed(4)}`);
    console.log(`  Dir to child: [${childDir.x.toFixed(4)}, ${childDir.y.toFixed(4)}, ${childDir.z.toFixed(4)}]`);
  }

  // Also dump raw bone names from the GLTF
  console.log("\n\n--- RAW BONE HIERARCHY FROM GLTF ---");
  function printBone(bone: THREE.Bone, depth: number) {
    const indent = "  ".repeat(depth);
    bone.updateWorldMatrix(true, true);
    const worldPos = new THREE.Vector3();
    const worldQuat = new THREE.Quaternion();
    bone.getWorldPosition(worldPos);
    bone.getWorldQuaternion(worldQuat);

    let childDir = new THREE.Vector3();
    if (bone.children.some(c => c instanceof THREE.Bone)) {
      const firstChild = bone.children.find(c => c instanceof THREE.Bone) as THREE.Bone;
      const childWorldPos = new THREE.Vector3();
      firstChild.getWorldPosition(childWorldPos);
      childDir.subVectors(childWorldPos, worldPos).normalize();
    }

    console.log(`${indent}${bone.name || "(unnamed)"}`);
    console.log(`${indent}  localPos:  [${bone.position.x.toFixed(4)}, ${bone.position.y.toFixed(4)}, ${bone.position.z.toFixed(4)}]`);
    console.log(`${indent}  localQuat: [${bone.quaternion.x.toFixed(4)}, ${bone.quaternion.y.toFixed(4)}, ${bone.quaternion.z.toFixed(4)}, ${bone.quaternion.w.toFixed(4)}]`);
    console.log(`${indent}  worldPos:  [${worldPos.x.toFixed(4)}, ${worldPos.y.toFixed(4)}, ${worldPos.z.toFixed(4)}]`);
    console.log(`${indent}  worldQuat: [${worldQuat.x.toFixed(4)}, ${worldQuat.y.toFixed(4)}, ${worldQuat.z.toFixed(4)}, ${worldQuat.w.toFixed(4)}]`);
    console.log(`${indent}  childDir:  [${childDir.x.toFixed(4)}, ${childDir.y.toFixed(4)}, ${childDir.z.toFixed(4)}]`);

    for (const child of bone.children) {
      if (child instanceof THREE.Bone) {
        printBone(child, depth + 1);
      }
    }
  }

  const skeleton = scene.getObjectByName("Root") || scene.getObjectByName("Hips") || scene.getObjectByName("mixamorigHips");
  if (skeleton instanceof THREE.Bone) {
    printBone(skeleton, 0);
  } else {
    // Try to find the first bone anywhere in the scene
    scene.traverse((obj) => {
      if (obj instanceof THREE.Bone && obj.parent instanceof THREE.Object3D && obj.parent.type !== "Bone") {
        console.log(`\nFound root bone: ${obj.name}`);
        printBone(obj, 0);
      }
    });
  }

  console.log("\n========== END REPORT ==========\n");
}

main().catch(console.error);
