/**
 * Test the ORIGINAL (un-normalized) tfjs_model_399 against the same
 * HELLO training window. If the original model recognizes HELLO and
 * the normalized one doesn't, the normalization script broke it.
 *
 * Usage: npx tsx scripts/verify-original-model.ts [label]
 */
import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs";
import { INFERENCE_LABELS } from "../src/inference/labels";

const TOTAL_FLOATS = 345;
const MODEL_DIR = path.resolve("training/model_out/tfjs_model_399");

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
  const labels = INFERENCE_LABELS;

  const label = process.argv[2] ?? "HELLO";
  const samplePath = `C:/Users/BRENT/AppData/Local/Temp/kilo/training-sample-${label}.json`;
  if (!fs.existsSync(samplePath)) {
    console.error(`no training sample at ${samplePath} — run extract-training-sample.mjs ${label} first`);
    process.exit(2);
  }
  const doc = JSON.parse(fs.readFileSync(samplePath, "utf8"));
  const frames = new Float32Array(doc.frames);

  const input = tf.tensor(frames, [1, 16, TOTAL_FLOATS]);
  const probs = model.predict(input) as tf.Tensor;
  const data = (await probs.data()) as Float32Array;
  input.dispose(); probs.dispose();

  const arr = Array.from(data);
  const ranked = arr.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  console.log(`ORIGINAL model on its own training window for ${label}:`);
  ranked.slice(0, 10).forEach((r, k) => {
    const mark = labels[r.i] === label ? "  <-- correct" : "";
    console.log(`  ${k + 1}. ${labels[r.i] ?? "?"} = ${r.p.toFixed(4)}${mark}`);
  });
  const correct = ranked.find((r) => labels[r.i] === label);
  console.log(`\ncorrect label rank: ${correct ? ranked.indexOf(correct) + 1 : "not in top 10"}  prob: ${correct ? correct.p.toFixed(4) : "n/a"}`);
  model.dispose();
}
main().catch((e) => { console.error("DRIVER FAILED:", e instanceof Error ? e.message : String(e)); process.exit(1); });