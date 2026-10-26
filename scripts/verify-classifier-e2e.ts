/**
 * End-to-end classifier smoke test: run a real canonical recording
 * through the SAME pipeline the live Translate view uses
 * (normalizeLandmarks → selectFeatures → sliceIntoWindows →
 * TemporalEnsemble) and report what the model reads.
 *
 * Run:  npx tsx scripts/verify-classifier-e2e.ts [label]
 * Defaults to HELLO, ADOPT, THANK_YOU, ARE_YOU_DEAF.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { sliceIntoWindows, WINDOW_SIZE, DEFAULT_STRIDE } from "../src/features/windowBuffer";
import { FEATURE_DIM } from "../src/features/featureSelect";
import { TemporalEnsemble } from "../src/inference/temporalEnsemble";
import { INFERENCE_LABELS } from "../src/inference/labels";

const ROOT = path.resolve("public/avatar-data");
const TOTAL_FLOATS = 345;
const MODEL_DIR = path.resolve("public/models/fsl_model");

function readCanonical(label: string): Float32Array[] | null {
  const urls = [
    path.join(ROOT, `${encodeURIComponent(label)}.json`),
    path.join(ROOT, `${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`),
  ];
  for (const url of urls) {
    if (!fs.existsSync(url)) continue;
    const doc = JSON.parse(fs.readFileSync(url, "utf-8")) as { frames: number[] };
    if (!Array.isArray(doc.frames)) continue;
    const flat = new Float32Array(doc.frames);
    const count = Math.floor(flat.length / TOTAL_FLOATS);
    const frames: Float32Array[] = [];
    for (let i = 0; i < count; i++) {
      frames.push(flat.slice(i * TOTAL_FLOATS, (i + 1) * TOTAL_FLOATS));
    }
    return frames;
  }
  return null;
}

async function main() {
  await tf.setBackend("cpu");
  await tf.ready();
  // Serve the local model files via data-URL fetch patching
  // (Node's fetch doesn't support file:// URLs).
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

  const targets = (process.argv.slice(2).length > 0 ? process.argv.slice(2) : ["HELLO", "ADOPT", "THANK_YOU", "ARE_YOU_DEAF"]);
  let passed = 0;
  let failed = 0;

  for (const label of targets) {
    const raw = readCanonical(label);
    if (!raw || raw.length === 0) {
      console.log(`FAIL  ${label} — no canonical recording`);
      failed++;
      continue;
    }
    // Canonical avatar-data files store 345-float FEATURE-SELECTED frames
    // (the capture studio writes selectFeatures(normalizeLandmarks(...))
    // output), so they feed the window buffer directly — no second
    // normalize/select pass, or the frame would be re-scaled.
    const feats = raw;
    const windows = sliceIntoWindows(feats, FEATURE_DIM, WINDOW_SIZE, DEFAULT_STRIDE);

    const ensemble = new TemporalEnsemble(labels.length, labels);
    const emissions: { label: string; confidence: number }[] = [];
    let debugFirst = true;
    for (const w of windows) {
      const probs = tf.tidy(() => {
        const input = tf.tensor(w, [1, WINDOW_SIZE, FEATURE_DIM]);
        return model.predict(input) as tf.Tensor;
      });
      const data = (await probs.data()) as Float32Array;
      probs.dispose();
      if (debugFirst) {
        // Print the raw top-5 probabilities so we can see whether the
        // model is producing sensible output at all.
        const arr = Array.from(data);
        const ranked = arr.map((p, i) => ({ p, i }))
          .sort((a, b) => b.p - a.p)
          .slice(0, 5);
        console.log(`  [debug] ${label} top-5: ` +
          ranked.map((r) => `${labels[r.i]}=${r.p.toFixed(3)}`).join("  "));
        debugFirst = false;
      }
      const result = ensemble.push(new Float32Array(data));
      if (result) emissions.push({ label: result.label, confidence: result.confidence });
    }

    const top = emissions[0] ?? null;
    const normalizedTarget = label.replace(/_/g, " ");
    const hit = top ? top.label === normalizedTarget : false;
    const detail = top ? `emitted=${top.label}(${top.confidence.toFixed(3)}) windows=${windows.length} emissions=${emissions.length}` : `no emission (windows=${windows.length})`;
    if (hit) { passed++; console.log(`PASS  ${label} — ${detail}`); }
    else { failed++; console.log(`FAIL  ${label} — ${detail}`); }
  }

  console.log(`\nCLASSIFIER E2E ${failed === 0 ? "PASSED" : "FAILED"}: ${passed} passed, ${failed} failed`);
  model.dispose();
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("DRIVER FAILED:", e instanceof Error ? e.message : String(e));
  if (e instanceof Error && e.stack) console.error("STACK:", e.stack);
  process.exit(1);
});