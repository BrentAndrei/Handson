/**
 * Extract ONE training sample of a given label from combined_dataset.json
 * WITHOUT loading the whole 5 GB file. JSON doesn't stream, so we scan
 * the file for the first `"label":"<LABEL>"` occurrence and parse just
 * the enclosing sample object by tracking brace depth from that point.
 *
 * Usage: node scripts/extract-training-sample.mjs <label> [outfile.json]
 */
import fs from "fs";

function main() {
  const label = process.argv[2];
  if (!label) { console.error("usage: extract-training-sample.mjs <label> [outfile]"); process.exit(2); }
  const src = "C:/Users/BRENT/Downloads/handson_fixed/handson/training/combined_dataset.json";
  const needle = `"label":"${label}"`;
  const fd = fs.openSync(src, "r");
  const buf = Buffer.alloc(1 << 20);
  let pos = 0;
  let foundAt = -1;
  let overlap = "";
  while (foundAt < 0) {
    const n = fs.readSync(fd, buf, 0, buf.length, pos);
    if (n === 0) break;
    const chunk = overlap + buf.toString("utf8", 0, n);
    const idx = chunk.indexOf(needle);
    if (idx >= 0) { foundAt = pos + idx - overlap.length; break; }
    overlap = chunk.slice(-needle.length);
    pos += n;
  }
  fs.closeSync(fd);
  if (foundAt < 0) { console.error(`label ${label} not found`); process.exit(1); }

  const fd2 = fs.openSync(src, "r");
  let start = foundAt;
  const back = Buffer.alloc(1 << 16);
  while (start > 0) {
    const readStart = Math.max(0, start - back.length);
    const n = fs.readSync(fd2, back, 0, back.length, readStart);
    const chunk = back.toString("utf8", 0, n);
    let i = chunk.length - 1;
    while (i >= 0 && chunk[i] !== "{") i--;
    if (i >= 0) { start = readStart + i; break; }
    start = readStart;
    if (n < back.length) break;
  }
  const forward = Buffer.alloc(1 << 22);
  const n = fs.readSync(fd2, forward, 0, forward.length, start);
  const text = forward.toString("utf8", 0, n);
  let depth = 0;
  let inStr = false;
  let esc = false;
  let end = -1;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) { esc = false; } else if (ch === "\\") { esc = true; } else if (ch === '"') { inStr = false; }
      continue;
    }
    if (ch === '"') { inStr = true; } else if (ch === "{") { depth++; } else if (ch === "}") { depth--; if (depth === 0) { end = i; break; } }
  }
  fs.closeSync(fd2);
  if (end < 0) { console.error("could not find end of sample"); process.exit(1); }
  const sample = text.slice(0, end + 1);
  const out = process.argv[3] || `C:/Users/BRENT/AppData/Local/Temp/kilo/training-sample-${label}.json`;
  fs.writeFileSync(out, sample);
  console.log(`wrote ${out} (${sample.length} bytes)`);
}
main();