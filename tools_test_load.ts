import * as fs from "node:fs";
import * as path from "node:path";
import * as tf from "@tensorflow/tfjs";

async function test() {
  await tf.setBackend("cpu");
  await tf.ready();
  
  const MODEL_DIR = path.resolve("public/models/fsl_model");
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
  
  try {
    console.log("Loading model...");
    const model = await tf.loadLayersModel("/models/fsl_model/model.json");
    console.log("Model loaded successfully!");
    console.log("Input shape:", model.inputs[0].shape);
    console.log("Output shape:", model.outputs[0].shape);
    model.dispose();
  } catch (e) {
    console.error("Error:", e);
    if (e instanceof Error) console.error("Stack:", e.stack);
  }
}

test().catch(console.error);