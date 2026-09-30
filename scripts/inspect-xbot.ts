import * as fs from "fs";

// Extract JSON chunk from GLB
const buf = fs.readFileSync("public/models/Xbot.glb");
const jsonStart = buf.indexOf("{");
const jsonEnd = buf.indexOf("}", jsonStart) + 1;
// The JSON chunk ends before the binary chunk - find the actual end
// GLB format: 12 byte header, then chunks
// Each chunk: 4 byte length, 4 byte type, then data
// JSON type = 0x4E4F534A

let offset = 12;
const firstChunkLen = buf.readUInt32LE(offset);
const firstChunkType = buf.toString("utf8", offset + 4, offset + 8);

if (firstChunkType === "JSON") {
  const jsonStr = buf.toString("utf8", offset + 8, offset + 8 + firstChunkLen);
  const json = JSON.parse(jsonStr);
  
  console.log("=== GLTF METADATA ===");
  console.log("asset:", JSON.stringify(json.asset));
  console.log("scene:", json.scene);
  console.log("scenes:", JSON.stringify(json.scenes));
  console.log("");
  
  console.log("=== NODES ===");
  console.log("total nodes:", json.nodes.length);
  for (let i = 0; i < json.nodes.length; i++) {
    const n = json.nodes[i];
    console.log(`[${i}] name="${n.name}" children=${JSON.stringify(n.children)} rotation=${JSON.stringify(n.rotation)} translation=${JSON.stringify(n.translation)} scale=${JSON.stringify(n.scale)} mesh=${n.mesh}`);
  }
  console.log("");
  
  console.log("=== SKINS ===");
  if (json.skins) {
    console.log("total skins:", json.skins.length);
    for (const skin of json.skins) {
      console.log("joints:", JSON.stringify(skin.joints));
      console.log("root:", skin.skeleton);
      if (skin.inverseBindMatrices) {
        console.log("inverseBindMatrices accessor:", skin.inverseBindMatrices);
      }
    }
  }
  console.log("");
  
  console.log("=== ACCESSORS (for bind matrices) ===");
  if (json.accessors) {
    for (const acc of json.accessors) {
      if (acc.type === "MAT4") {
        console.log("MAT4 accessor:", JSON.stringify({ bufferView: acc.bufferView, count: acc.count, type: acc.type }));
      }
    }
  }
  
  console.log("");
  console.log("=== ANIMATIONS ===");
  if (json.animations) {
    console.log("total animations:", json.animations.length);
    for (const anim of json.animations) {
      console.log("animation:", anim.name, "channels:", anim.channels?.length, "samplers:", anim.samplers?.length);
    }
  }
} else {
  console.log("First chunk type:", firstChunkType);
}
