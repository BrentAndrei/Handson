const fs = require('fs');
const data = fs.readFileSync('public/models/Xbot.glb');
// GLB: 12-byte header (magic 4, length 4, version 4), then chunks
const jsonLength = data.readUInt32LE(12);
const jsonStr = data.slice(20, 20 + jsonLength).toString('utf8');
const gltf = JSON.parse(jsonStr);

// Collect bone names from nodes
const boneNames = [];
const seen = new Set();
function traverse(nodes) {
  if (!nodes) return;
  for (const node of nodes) {
    if (node.name && !seen.has(node.name)) {
      seen.add(node.name);
      boneNames.push(node.name);
    }
    if (node.children) traverse(node.children);
  }
}
if (gltf.scenes) {
  for (const scene of gltf.scenes) {
    if (scene.nodes) traverse(scene.nodes);
  }
}

// Check which nodes are likely bones (have transform or are referenced by skins)
const skinRefs = new Set();
if (gltf.skins) {
  for (const skin of gltf.skins) {
    if (skin.joints) {
      // Joints are referenced by index into nodes array - but we need the scene graph traversal
    }
    if (skin.skeleton !== undefined && gltf.nodes && gltf.nodes[skin.skeleton]) {
      // Root bone node
    }
  }
}

// Mixamo bones typically have mixamorig: prefix
const mixamo = boneNames.filter(n => n.includes('mixamorig') || n.includes('Mixamo') || n === 'Hips' || n === 'Spine');
console.log('Potential Mixamo bones:', JSON.stringify(mixamo, null, 1));
console.log('All bone names:', JSON.stringify(boneNames, null, 1));
