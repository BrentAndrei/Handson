/**
 * Checks the neutral_2 fixture's right middle finger: its curl,
 * and how far the DIP->TIP segment direction is from the
 * straight MCP->TIP line. If the finger is marginally bent
 * (curl just under 0.02) the segment directions legitimately
 * differ from the straight line by a few degrees, which is
 * what the flat-hand -dir gate sees.
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { mediapipeToCanonicalPose } from "../src/avatar/mediapipeAdapter";
import type { Vec3 } from "../src/avatar/canonicalPose";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = path.join(__dirname, "fixtures", "poses", "extracted");

const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l < 1e-10 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
};
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const ang = (a: Vec3, b: Vec3) =>
  (Math.acos(Math.min(1, Math.max(-1, dot(norm(a), norm(b))))) * 180) / Math.PI;

function main() {
  const which = process.argv[2] ?? "neutral_2.bin";
  const raw = fs.readFileSync(path.join(FIXTURE_DIR, which));
  const v = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const pose = mediapipeToCanonicalPose(v, 0)!;
  const hand = pose.rightHand!;
  const fp = hand.fingers.middle;

  console.log(`${which} right middle: curl=${fp.curl.toFixed(4)}`);
  console.log(`  fp.dir (MCP->TIP): [${fp.dir.map((n) => n.toFixed(3))}]`);
  if (fp.jointDirs) {
    const segs = ["MCP->PIP", "PIP->DIP", "DIP->TIP"];
    fp.jointDirs.forEach((jd, i) => {
      const a = ang(jd, fp.dir);
      console.log(`  jointDirs[${i}] (${segs[i]}): off fp.dir by ${a.toFixed(2)}deg`);
    });
  }
}

main();
