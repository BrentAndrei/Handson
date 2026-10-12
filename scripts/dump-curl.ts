/**
 * Dumps the per-finger curl for a recording, to check whether
 * the source landmark data is reasonable for the sign.
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const FRAME_FLOATS = 345;

const FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;

function main() {
  const recordings = process.argv.slice(2).length > 0 ? process.argv.slice(2) : ["A", "D", "O", "P", "T"];

  for (const name of recordings) {
    const file = path.join(DATA_DIR, `${name}.json`);
    if (!fs.existsSync(file)) { console.warn(`skip ${name}`); continue; }
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    const flat: number[] = Array.isArray(parsed) ? parsed.flat() : parsed.frames;
    const frameCount = Math.floor(flat.length / FRAME_FLOATS);

    let framesWithHand = 0;
    const curlSum: Record<string, number> = {};
    const bendSum: Record<string, number> = {};
    for (const f of FINGERS) { curlSum[f] = 0; bendSum[f] = 0; }

    for (let fi = 0; fi < frameCount; fi++) {
      const frame = flat.slice(fi * FRAME_FLOATS, (fi + 1) * FRAME_FLOATS);
      const pose = mediapipeToCanonicalPose(frame, fi / 24);
      if (!pose) continue;
      const hand = pose.rightHand ?? pose.leftHand;
      if (!hand) continue;
      framesWithHand++;
      for (const f of FINGERS) {
        const fp = hand.fingers[f];
        if (!fp) continue;
        curlSum[f] += fp.curl;
        if (fp.bendDir) bendSum[f] += 1;
      }
    }

    console.log(`\n${name} (${frameCount} frames, ${framesWithHand} with a hand):`);
    for (const f of FINGERS) {
      const meanCurl = framesWithHand > 0 ? curlSum[f] / framesWithHand : 0;
      const bendFrac = framesWithHand > 0 ? bendSum[f] / framesWithHand : 0;
      console.log(`  ${f.padEnd(7)} mean curl ${meanCurl.toFixed(3)}   bendDir present ${(100 * bendFrac).toFixed(0)}%`);
    }
  }
}

main();
