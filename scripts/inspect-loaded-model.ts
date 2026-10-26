/**
 * Load the normalized model and print every layer name + weight name +
 * weight shape, so we can see whether the manifest names line up with
 * the layers tfjs actually constructs. A silent weight mis-assignment
 * (right shapes, wrong layers) produces near-uniform garbage output.
 *
 * Usage: npx tsx scripts/inspect-loaded-model.ts
 */
import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs";

const MODEL_DIR = path.resolve("public/models/fsl_model");

async function main() {
  await tf.setBackend("cpu");
  await tf.ready();
  const modelJson = fs.readFileSync(path.join(MODEL_DIR, "model.json"), "utf8");
  const binBuf = fs.readFileSync(path.join(MODEL_DIR, "group1-shard1of1.bin"));
  const origFetch = globalThis.fetch.bind(globalThis);
  const jsonUrl = `data:application/json;base64,${Buffer.from(modelJson).toString("base64")}`;
  const binUrl = `data:application/octet-stream;base64,${Buffer.from(binBuf).toString("base64")}`;
  (globalThis as any).fetch = (input: any, init?: any) => {
    const url = typeof input === "string" ? input : input?.url;
    if (url && url.endsWith("model.json")) return fetch(jsonUrl);
    if (url && (url.endsWith(".bin") || url.includes("group1-shard1of1"))) return fetch(binUrl);
    return origFetch(input, init);
  };
  const model = await tf.loadLayersModel("/models/fsl_model/model.json");
  console.log("=== Layers ===");
  for (const l of model.layers) {
    const ws = l.getWeights();
    console.log(`  ${(l as any).name ?? l.constructor.name} (${l.constructor.name}) weights=${ws.length}`);
    for (const w of ws) console.log(`      ${(w as any).name ?? "weight"} shape=${JSON.stringify((w as any).shape)}`);
  }
  console.log("\n=== Manifest names ===");
  const manifest = JSON.parse(modelJson).weightsManifest[0].weights;
  for (const w of manifest) console.log(`  ${(w as any).name} shape=${JSON.stringify((w as any).shape)}`);

  // Check for unmatched weights
  const loadedNames = new Set<string>();
  for (const l of model.layers) for (const w of l.getWeights()) loadedNames.add((w as any).name);
  const manifestNames = new Set<string>(manifest.map((w: any) => w.name));
  const unmatched = [...manifestNames].filter((n: string) => !loadedNames.has(n));
  const extra = [...loadedNames].filter((n) => !manifestNames.has(n));
  console.log(`\nmanifest weights not on any layer: ${unmatched.length}`);
  for (const n of unmatched.slice(0, 5)) console.log(`  MISSING: ${n}`);
  console.log(`layer weights not in manifest: ${extra.length}`);
  for (const n of extra.slice(0, 5)) console.log(`  EXTRA: ${n}`);
  model.dispose();
}
main().catch((e) => { console.error("DRIVER FAILED:", e instanceof Error ? e.message : String(e)); process.exit(1); });