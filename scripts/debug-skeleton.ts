import * as fs from "fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { resolveSkeleton } from "../src/avatar/boneResolver";

async function main() {
  const output: string[] = [];
  const log = (...args: any[]) => { output.push(args.map(String).join(" ")); };

  const buf = fs.readFileSync("public/models/Xbot.glb");
  log("buffer length:", buf.length);

  const loader = new GLTFLoader();
  const gltf: any = await new Promise((resolve, reject) => {
    loader.parse(buf, "", resolve, undefined, reject);
  });

  log("GLTF loaded!");
  const resolved = resolveSkeleton(gltf.scene);
  log("bones count:", resolved.bones.size);

  const roles = ["hips", "spine1", "spine2", "spine3", "neck", "head",
    "leftShoulder", "leftUpperArm", "leftForearm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightForearm", "rightHand"];

  for (const role of roles) {
    const bone = resolved.getBone(role as any);
    if (bone) {
      log("\n--- " + role + " => " + bone.name + " ---");
      log("  parent:", bone.parent);
      log("  children:", JSON.stringify(bone.children));
      log("  localPos:", JSON.stringify(bone.localPos.toArray().map(n => n.toFixed(4))));
      log("  localQuat:", JSON.stringify(bone.localRot.toArray().map(n => n.toFixed(4))));
      log("  worldPos:", JSON.stringify(bone.worldPos.toArray().map(n => n.toFixed(4))));
      log("  worldQuat:", JSON.stringify(bone.worldRot.toArray().map(n => n.toFixed(4))));
      log("  localDir:", JSON.stringify(bone.localDir.toArray().map(n => n.toFixed(4))));
      log("  length:", bone.length.toFixed(4));
    } else {
      log("MISSING:", role);
    }
  }

  // Finger bones
  log("\n--- FINGERS ---");
  for (const key of Object.keys(resolved.boneMapping)) {
    if (key.includes("Finger")) {
      const bone = resolved.bones.get(resolved.boneMapping[key]);
      if (bone) {
        log(key, "=>", bone.name, "localDir:", JSON.stringify(bone.localDir.toArray().map((n: number) => n.toFixed(4))));
      }
    }
  }

  log("\nrootBone:", resolved.rootBone);
  log("armatureScale:", resolved.armatureScale);
  log("boneMapping:", JSON.stringify(resolved.boneMapping, null, 2));

  fs.writeFileSync("skeleton-report.txt", output.join("\n"));
  console.log("Report written to skeleton-report.txt");
  console.log("\n" + output.join("\n"));
}

main().catch(e => {
  console.error("ERROR:", e);
  fs.writeFileSync("skeleton-report.txt", "ERROR: " + String(e));
});
