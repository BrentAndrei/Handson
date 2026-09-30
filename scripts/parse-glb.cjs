const fs = require("fs");
const path = require("path");

const buf = fs.readFileSync("public/models/Xbot.glb");
const chunkLen = buf.readUInt32LE(12);
const jsonStr = buf.toString("utf8", 20, 20 + chunkLen);
const json = JSON.parse(jsonStr);

const output = [];

// Build node lookup
const nodes = json.nodes.map((n, i) => ({ ...n, idx: i }));

// Build parent map
const parentMap = {};
for (let i = 0; i < nodes.length; i++) {
  const n = nodes[i];
  if (n.children) {
    for (const c of n.children) {
      parentMap[c] = i;
    }
  }
}

// Find root (the node that's not a child of anyone)
const rootIdx = json.scenes[json.scene].nodes[0];

output.push("=== FULL SKELETON REPORT ===\n");

// Print complete hierarchy
output.push("--- BONE HIERARCHY ---");
function printBone(idx, depth) {
  const n = nodes[idx];
  if (!n) return;
  const indent = "  ".repeat(depth);
  const name = n.name || "(unnamed)";
  const localPos = n.translation || [0, 0, 0];
  const localRot = n.rotation || [0, 0, 0, 1];
  
  // Compute child direction
  let childDir = "[no children]";
  let boneLen = 0;
  if (n.children && n.children.length > 0) {
    const childIdx = n.children[0];
    const child = nodes[childIdx];
    if (child && child.translation) {
      const mag = Math.sqrt(child.translation.reduce((s, v) => s + v*v, 0));
      childDir = "[" + child.translation.map(v => (v/mag).toFixed(4)).join(", ") + "]";
      boneLen = mag;
    }
  }
  
  // Normalize localPos for direction
  const posMag = Math.sqrt(localPos.reduce((s, v) => s + v*v, 0));
  const localDir = posMag > 0.001 ? localPos.map(v => (v/posMag).toFixed(4)) : ["0.0000", "0.0000", "0.0000"];
  
  const rotStr = localRot[0] < 1e-6 && localRot[1] < 1e-6 && localRot[2] < 1e-6 
    ? "[identity]" : "[" + localRot.map(v => v.toFixed(8)).join(", ") + "]";
  
  output.push(indent + name);
  output.push(indent + "  localPos:  [" + localPos.map(v => v.toFixed(4)).join(", ") + "]");
  output.push(indent + "  localDir:  [" + localDir.join(", ") + "]");
  output.push(indent + "  localQuat: " + rotStr);
  output.push(indent + "  parent:    " + (parentMap[idx] !== undefined ? nodes[parentMap[idx]].name : "(root)"));
  output.push(indent + "  children:  " + (n.children ? n.children.map(c => nodes[c].name).join(", ") : "[]"));
  output.push(indent + "  boneLen:   " + boneLen.toFixed(4));
  output.push("");
  
  if (n.children) {
    for (const c of n.children) {
      printBone(c, depth + 1);
    }
  }
}

printBone(rootIdx, 0);

// Print key bone analysis
output.push("\n--- KEY BONE ANALYSIS (GLTF localDir, converted to canonical) ---");
const gltfToCanonical = (v) => [v[2], v[1], v[0]];
const canonicalToGltf = (v) => [v[2], v[1], v[0]];

const keyBones = [
  ["Spine", "Hips -> Spine"],
  ["Spine1", "Spine -> Spine1"],
  ["Spine2", "Spine1 -> Spine2"],
  ["Neck", "Spine2 -> Neck"],
  ["Head", "Neck -> Head"],
  ["LeftShoulder", "Spine2 -> LeftShoulder"],
  ["LeftArm", "LeftShoulder -> LeftArm"],
  ["LeftForeArm", "LeftArm -> LeftForeArm"],
  ["LeftHand", "LeftForeArm -> LeftHand"],
  ["RightShoulder", "Spine2 -> RightShoulder"],
  ["RightArm", "RightShoulder -> RightArm"],
  ["RightForeArm", "RightArm -> RightForeArm"],
  ["RightHand", "RightForeArm -> RightHand"],
  ["LeftUpLeg", "Hips -> LeftUpLeg"],
  ["LeftLeg", "LeftUpLeg -> LeftLeg"],
  ["LeftFoot", "LeftLeg -> LeftFoot"],
  ["RightUpLeg", "Hips -> RightUpLeg"],
  ["RightLeg", "RightUpLeg -> RightLeg"],
  ["RightFoot", "RightLeg -> RightFoot"],
];

for (const [boneName, parent] of keyBones) {
  const fullName = "mixamorig:" + boneName;
  const node = nodes.find(n => n.name === fullName);
  if (node) {
    const localPos = node.translation || [0, 0, 0];
    const mag = Math.sqrt(localPos.reduce((s, v) => s + v*v, 0));
    const norm = localPos.map(v => v/mag);
    const canon = gltfToCanonical(norm);
    
    // Determine primary axis
    const axes = ["X", "Y", "Z"];
    let primary = "?";
    let maxVal = 0;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(canon[i]) > maxVal) {
        maxVal = Math.abs(canon[i]);
        primary = (canon[i] > 0 ? "+" : "-") + axes[i];
      }
    }
    
    output.push(boneName + " (" + parent + ")");
    output.push("  GLTF localDir:   [" + norm.map(v => v.toFixed(4)).join(", ") + "]");
    output.push("  Canonical dir:   [" + canon.map(v => v.toFixed(4)).join(", ") + "]");
    output.push("  Primary axis:    " + primary + " (" + (maxVal*100).toFixed(1) + "%)");
    output.push("  Bone length:     " + mag.toFixed(4) + " cm");
    output.push("");
  }
}

// CanonICAL coordinate reference
output.push("\n--- COORDINATE SYSTEM REFERENCE ---");
output.push("GLTF/Three.js: X=right, Y=up, Z=out(forward)");
output.push("Canonical:    X=forward, Y=up, Z=right");
output.push("gltfToCanonical: [x,y,z] -> [z,y,x]");
output.push("");
output.push("Armature scale: 0.01 (cm to meters)");
output.push("Hips position (scaled): [0, " + (103.99 * 0.01).toFixed(2) + ", " + (2.08 * 0.01).toFixed(2) + "] m");

// Bone mapping
output.push("\n--- BONE MAPPING ---");
const mapping = {
  "hips": "mixamorig:Hips",
  "spine1": "mixamorig:Spine",
  "spine2": "mixamorig:Spine1",
  "spine3": "mixamorig:Spine2",
  "neck": "mixamorig:Neck",
  "head": "mixamorig:Head",
  "leftShoulder": "mixamorig:LeftShoulder",
  "leftUpperArm": "mixamorig:LeftArm",
  "leftForearm": "mixamorig:LeftForeArm",
  "leftHand": "mixamorig:LeftHand",
  "rightShoulder": "mixamorig:RightShoulder",
  "rightUpperArm": "mixamorig:RightArm",
  "rightForearm": "mixamorig:RightForeArm",
  "rightHand": "mixamorig:RightHand",
  "leftHip": "mixamorig:LeftUpLeg",
  "leftKnee": "mixamorig:LeftLeg",
  "leftAnkle": "mixamorig:LeftFoot",
  "rightHip": "mixamorig:RightUpLeg",
  "rightKnee": "mixamorig:RightLeg",
  "rightAnkle": "mixamorig:RightFoot",
};

for (const [role, glbName] of Object.entries(mapping)) {
  const node = nodes.find(n => n.name === glbName);
  if (node) {
    const localPos = node.translation || [0, 0, 0];
    const mag = Math.sqrt(localPos.reduce((s, v) => s + v*v, 0));
    const norm = localPos.map(v => v/mag);
    const canon = gltfToCanonical(norm);
    output.push(role + " -> " + glbName + " localDir=" + JSON.stringify(norm.map(v => v.toFixed(4))) + " canonDir=" + JSON.stringify(canon.map(v => v.toFixed(4))));
  }
}

fs.writeFileSync("skeleton-report.txt", output.join("\n"));
