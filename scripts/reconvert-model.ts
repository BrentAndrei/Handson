/**
 * Re-convert the 399-class model with the CURRENT tfjs-converter (v4.22.0)
 * so it is loadable by the installed @tensorflow/tfjs (also 4.22.0).
 *
 * The existing tfjs_model_399 was produced by converter v3.18.0 and fails to
 * load in tfjs 4.22.0 with "Failed to parse model JSON".
 *
 * Uses tfjs-converter's node entry point directly (no tfjs-backend-node needed).
 *
 * Run:  npx tsx scripts/reconvert-model.ts
 */
// convertTfLayersModel removed from tfjs-converter v4.22.0
// Run manual conversion via: tensorflowjs_converter --input_format=tf_saved_model --output_format=keras_saved_model
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const SAVED = path.join(ROOT, "training", "model_out", "saved_model");

async function main() {
  if (!fs.existsSync(SAVED)) {
    throw new Error("Saved model not found at " + SAVED);
  }
  throw new Error("reconvert-model.ts: convertTfLayersModel no longer available in tfjs-converter v4.22.0. Model should be converted via tensorflowjs_converter CLI.");
}

main().catch((e) => {
  console.error("FAILED:", e instanceof Error ? e.message : String(e));
  if (e instanceof Error && e.stack) console.error(e.stack);
  process.exit(1);
});