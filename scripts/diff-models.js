/**
 * Deep-diff two model.json files to find the remaining structural difference
 * after normalization.
 */
import * as fs from "fs";

function canon(v) {
  if (v === null || v === undefined) return v;
  if (typeof v !== "object") return v;
  if (Array.isArray(v)) return v.map(canon);
  const o = {};
  for (const k of Object.keys(v).sort()) o[k] = canon(v[k]);
  return o;
}

const oldPath = process.argv[2];
const newPath = process.argv[3];
if (!oldPath || !newPath) {
  console.error("Usage: node scripts/diff-models.js <old.json> <new.json>");
  process.exit(2);
}
const a = JSON.parse(fs.readFileSync(oldPath, "utf8"));
const b = JSON.parse(fs.readFileSync(newPath, "utf8"));

const ca = canon(a.modelTopology.model_config);
const cb = canon(b.modelTopology.model_config);
const sa = JSON.stringify(ca);
const sb = JSON.stringify(cb);
console.log("model_config identical:", sa === sb);
if (sa !== sb) {
  // find first differing index
  let i = 0;
  while (i < Math.min(sa.length, sb.length) && sa[i] === sb[i]) i++;
  console.log("first diff at char", i);
  console.log("OLD:", sa.slice(Math.max(0, i - 80), i + 200));
  console.log("NEW:", sb.slice(Math.max(0, i - 80), i + 200));
}