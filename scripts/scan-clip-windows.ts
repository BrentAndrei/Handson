/**
 * Scan every sliding window of a canonical clip and report the MAX
 * probability the model assigns to the correct label, plus the max
 * top-1 probability overall. A window in the middle of the sign
 * (where the gesture is actually happening) should score much higher
 * than the first/last windows.
 *
 * Usage: npx tsx scripts/scan-clip-windows.ts [label]
 */
import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs";
import { sliceIntoWindows, WINDOW_SIZE, DEFAULT_STRIDE } from "../src/features/windowBuffer";
import { FEATURE_DIM } from "../src/features/featureSelect";
import { INFERENCE_LABELS } from "../src/inference/labels";

const TOTAL_FLOATS = 345;
const ROOT = path.resolve("public/avatar-data");
const MODEL_DIR = path.resolve("public/models/fsl_model");

function readCanonical(label: string): Float32Array[] | null {
  const urls = [
    path.join(ROOT, `${encodeURIComponent(label)}.json`),
    path.join(ROOT, `${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`),
  ];
  for (const url of urls) {
    if (!fs.existsSync(url)) continue;
    const doc = JSON.parse(fs.readFileSync(url, "utf8")) as { frames: number[] };
    if (!Array.isArray(doc.frames)) continue;
    const flat = new Float32Array(doc.frames);
    const count = Math.floor(flat.length / TOTAL_FLOATS);
    const frames: Float32Array[] = [];
    for (let i = 0; i < count; i++) frames.push(flat.slice(i * TOTAL_FLOATS, (i + 1) * TOTAL_FLOATS));
    return frames;
  }
  return null;
}

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
  const raw = readCanonical(label);
  if (!raw) { console.error(`no canonical for ${label}`); process.exit(2); }
  const windows = sliceIntoWindows(raw, FEATURE_DIM, WINDOW_SIZE, DEFAULT_STRIDE);
  console.log(`${label}: ${raw.length} frames -> ${windows.length} windows`);

  let maxCorrect = 0;
  let maxTop1 = 0;
  let maxTop1Label = "";
  let bestWindow = -1;
  for (let w = 0; w < windows.length; w++) {
    const probs = tf.tidy(() => {
      const input = tf.tensor(windows[w], [1, WINDOW_SIZE, FEATURE_DIM]);
      return model.predict(input) as tf.Tensor;
    });
    const data = (await probs.data()) as Float32Array;
    probs.dispose();
    const arr = Array.from(data);
    let top1 = 0, top1Idx = 0;
    for (let i = 0; i < arr.length; i++) { if (arr[i] > arr[top1]) { top1 = arr[i]; top1Idx = i; } }
    const correctIdx = labels.indexOf(label);
    const correct = arr[correctIdx];
    if (correct > maxCorrect) { maxCorrect = correct; bestWindow = w; }
    if (top1 > maxTop1) { maxTop1 = top1; maxTop1Label = labels[top1Idx] ?? "?"; }
  }
  console.log(`max prob for ${label}: ${maxCorrect.toFixed(4)} (window ${bestWindow})`);
  console.log(`max top-1 prob: ${maxTop1.toFixed(4)} = ${maxTop1Label}`);
  console.log(`${label} index in labels: ${labels.indexOf(label)}`);
  model.dispose();
}
main().catch((e) => { console.error("DRIVER FAILED:", e instanceof Error ? e.message : String(e)); process.exit(1); });