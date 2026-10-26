/**
 * Run the deployed model on ONE real training window (16 frames) and
 * print the top-10 probabilities. Decisive test: if the model can't
 * even recognize its own training sample, the model/weights are
 * broken; if it can, the ensemble thresholds are too strict.
 *
 * Usage: npx tsx scripts/verify-model-on-training-window.ts [label]
 */
import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs";
import { INFERENCE_LABELS } from "../src/inference/labels";

const TOTAL_FLOATS = 345;
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
  const labels = INFERENCE_LABELS;

  const targets = (process.argv.slice(2).length > 0 ? process.argv.slice(2) : ["HELLO", "A", "B", "C"]);
  let passed = 0, failed = 0;
  for (const label of targets) {
    const samplePath = `C:/Users/BRENT/AppData/Local/Temp/kilo/training-sample-${label}.json`;
    if (!fs.existsSync(samplePath)) {
      console.log(`SKIP  ${label} — no training sample (run extract-training-sample.mjs ${label} first)`);
      continue;
    }
    const doc = JSON.parse(fs.readFileSync(samplePath, "utf8"));
    const frames = new Float32Array(doc.frames);
    if (frames.length !== 16 * TOTAL_FLOATS) {
      console.log(`SKIP  ${label} — sample has ${frames.length} floats, expected ${16 * TOTAL_FLOATS}`);
      continue;
    }

    const input = tf.tensor(frames, [1, 16, TOTAL_FLOATS]);
    const probs = model.predict(input) as tf.Tensor;
    const data = (await probs.data()) as Float32Array;
    input.dispose(); probs.dispose();

    const arr = Array.from(data);
    const ranked = arr.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
    const correct = ranked.find((r) => labels[r.i] === label);
    const rank = correct ? ranked.indexOf(correct) + 1 : -1;
    const prob = correct ? correct.p : 0;
    const hit = rank === 1;
    if (hit) { passed++; console.log(`PASS  ${label} — rank 1 prob ${prob.toFixed(4)}`); }
    else { failed++; console.log(`FAIL  ${label} — rank ${rank} prob ${prob.toFixed(4)}  top1=${labels[ranked[0].i]}(${ranked[0].p.toFixed(4)})`); }
  }
  console.log(`\nMODEL-ON-TRAINING ${failed === 0 ? "PASSED" : "FAILED"}: ${passed} passed, ${failed} failed`);
  model.dispose();
  if (failed > 0) process.exit(1);
}
main().catch((e) => { console.error("DRIVER FAILED:", e instanceof Error ? e.message : String(e)); process.exit(1); });