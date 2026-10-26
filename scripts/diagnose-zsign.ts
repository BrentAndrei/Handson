import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { mediapipeToCanonical } from "../src/avatar/coordinateSystem";
import type { Vec3 } from "../src/avatar/canonicalPose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, "..", "public", "avatar-data");
const FF = 345;

for (const rec of ["A", "D", "O", "P", "T", "HELLO", "ADOPT"]) {
  const file = path.join(DATA_DIR, `${rec}.json`);
  if (!fs.existsSync(file)) { console.log(`!! ${rec} not found`); continue; }
  const j = JSON.parse(fs.readFileSync(file, "utf8")) as { frames: number[] };
  const flat = new Float32Array(j.frames);
  const n = Math.floor(flat.length / FF);
  // pose shoulders and wrists for each frame, find the max hand presence
  let bestFi = -1, bestSpread = -1;
  for (let i = 0; i < n; i++) {
    const f = flat.slice(i * FF, (i + 1) * FF);
    const sp = Math.abs(f[99 * 3]) +
               Math.abs(f[162 * 3]);
    void sp;
    const lSpread = Math.abs(f[99 * 3] - f[102 * 3]);
    if (lSpread > bestSpread) { bestSpread = lSpread; bestFi = i; }
  }
  if (bestFi < 0) { console.log(`${rec}: no frames`); continue; }
  const f = flat.slice(bestFi * FF, (bestFi + 1) * FF);

  // anchor = midpoint of shoulders (raw)
  const ls = [f[11 * 3], f[11 * 3 + 1], f[11 * 3 + 2]] as Vec3;
  const rs = [f[12 * 3], f[12 * 3 + 1], f[12 * 3 + 2]] as Vec3;
  const anchor = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2] as Vec3;
  const cLw = mediapipeToCanonical([f[15 * 3], f[15 * 3 + 1], f[15 * 3 + 2]]);
  const cRw = mediapipeToCanonical([f[16 * 3], f[16 * 3 + 1], f[16 * 3 + 2]]);
  const cSlot99 = mediapipeToCanonical([f[99 * 3], f[99 * 3 + 1], f[99 * 3 + 2]]);
  const cSlot162 = mediapipeToCanonical([f[162 * 3], f[162 * 3 + 1], f[162 * 3 + 2]]);

  console.log(`\n=== ${rec} frame ${bestFi} ===`);
  console.log(`  raw shoulders: L(11) x=${ls[0].toFixed(3)}  R(12) x=${rs[0].toFixed(3)}  anchor x=${anchor[0].toFixed(3)}`);
  console.log(`  canon wrist16(LW) Z=${cLw[2].toFixed(3)}  wrist17(RW) Z=${cRw[2].toFixed(3)}`);
  console.log(`  canon slot99  Z=${cSlot99[2].toFixed(3)}  slot162 Z=${cSlot162[2].toFixed(3)}`);
  console.log(`  pose LW Y=${cLw[1].toFixed(3)} (up)  pose RW Y=${cRw[1].toFixed(3)}`);
  console.log(`  slot99 Y=${cSlot99[1].toFixed(3)}  slot162 Y=${cSlot162[1].toFixed(3)}`);
}
