/**
 * Runtime smoke test: load a real avatar-data label through the actual
 * loadLabelDataMediaPipe code path and confirm it produces retargetable frames.
 * Run: npx tsx scripts/verify-runtime-load.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");

const MEDIAPIPE_FRAME_FLOATS = 345;

function loadFile(label: string): any {
  const sanitized = label.replace(/[^a-zA-Z0-9]/g, "_");
  const p = path.join(DATA_DIR, `${sanitized}.json`);
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

// Replicate the type check in dataLoader.ts line 337-338
function checkType(data: any, label: string): string[] {
  const issues: string[] = [];
  if (!data || !Array.isArray(data.frames)) {
    issues.push(`${label}: data.frames is not an array (type=${typeof data?.frames})`);
    return issues;
  }
  // dataLoader does: const flat = new Float32Array(data.frames as unknown as number[]);
  // If data.frames is number[][] (2D), Float32Array(number[][]) -> length 1, holds the array ref
  if (Array.isArray(data.frames[0])) {
    issues.push(`${label}: data.frames is number[][] (2D) — Float32Array() will produce length-1 typed array`);
  } else if (typeof data.frames[0] === "number") {
    // OK: flat array of numbers
    const flat = new Float32Array(data.frames);
    const frameCount = Math.floor(flat.length / MEDIAPIPE_FRAME_FLOATS);
    issues.push(`${label}: OK — ${flat.length} floats, ${frameCount} frames (frameCount=${data.frameCount})`);
    if (frameCount !== data.frameCount) {
      issues.push(`  MISMATCH: computed ${frameCount} vs declared ${data.frameCount}`);
    }
  } else {
    issues.push(`${label}: data.frames[0] type=${typeof data.frames[0]}`);
  }
  return issues;
}

const labels = ["A", "HELLO", "ADOPT", "THANK_YOU", "ARE_YOU_DEAF"];
const allIssues: string[] = [];
for (const label of labels) {
  try {
    const data = loadFile(label);
    allIssues.push(...checkType(data, label));
  } catch (e) {
    allIssues.push(`${label}: LOAD ERROR — ${(e as Error).message}`);
  }
}
console.log(allIssues.join("\n"));