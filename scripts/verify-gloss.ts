/**
 * Gate for the englishToGloss fingerspelling fallback.
 *
 * Asserts:
 *   G1  known words still map to their dictionary gloss
 *   G2  OOV pure-alpha words decompose into single-letter glosses
 *   G3  every decomposed letter is flagged fingerspelled with role unknown
 *   G4  words containing digits/symbols are NOT fingerspelled (no digit signs)
 *   G5  single-letter OOV word ("a") fingerspells to "A"
 *   G6  dictionary single letter ("i") still maps to "I", not fingerspelled
 *   G7  trace records the fingerspell decomposition
 *   G8  phrase lookup still wins over word/letter lookup
 *   G9  all 26 letter glosses exist in the real model vocabulary manifest
 *  G10  empty input yields no glosses
 */
import * as fs from "node:fs";
import { englishToGloss } from "../src/features/englishToGloss";

let checks = 0;
let failures = 0;

function check(name: string, ok: boolean, detail = ""): void {
  checks++;
  if (!ok) {
    failures++;
    console.log(`  FAIL ${name}${detail ? `: ${detail}` : ""}`);
  } else {
    console.log(`  ok   ${name}`);
  }
}

// G1: dictionary hit
{
  const r = englishToGloss("hello");
  check("G1 known word maps to gloss",
    r.glosses.length === 1 && r.glosses[0].gloss === "HELLO" && !r.glosses[0].fingerspelled,
    JSON.stringify(r.glosses));
}

// G2: OOV word fingerspells letter-by-letter
{
  const r = englishToGloss("kumusta");
  check("G2 OOV word decomposes to letters",
    r.glosses.length === 7 && r.glosses.map((g) => g.gloss).join("") === "KUMUSTA",
    `got ${r.glosses.length} tokens: ${r.glosses.map((g) => g.gloss).join(",")}`);
}

// G3: decomposed tokens carry the fingerspelled marker
{
  const r = englishToGloss("kumusta");
  check("G3 letters flagged fingerspelled/unknown",
    r.glosses.every((g) => g.fingerspelled === true && g.role === "unknown" && g.meaning.startsWith("letter ")),
    JSON.stringify(r.glosses[0]));
}

// G4: digits block fingerspelling (no recorded digit signs)
{
  const r = englishToGloss("wifi5");
  check("G4 digit words stay single unknown token",
    r.glosses.length === 1 && r.glosses[0].gloss === "WIFI5" && !r.glosses[0].fingerspelled,
    JSON.stringify(r.glosses));
}

// G5: single-letter OOV word
{
  const r = englishToGloss("a");
  check("G5 'a' fingerspells to letter A",
    r.glosses.length === 1 && r.glosses[0].gloss === "A" && r.glosses[0].fingerspelled === true,
    JSON.stringify(r.glosses));
}

// G6: dictionary single letter wins over fingerspelling
{
  const r = englishToGloss("i");
  check("G6 'i' maps to dictionary sign I",
    r.glosses.length === 1 && r.glosses[0].gloss === "I" && !r.glosses[0].fingerspelled,
    JSON.stringify(r.glosses));
}

// G7: trace records the decomposition
{
  const r = englishToGloss("kumusta");
  check("G7 trace records fingerspell",
    r.trace.some((t) => t.startsWith("fingerspell: \"kumusta\"")),
    r.trace.join(" | "));
}

// G8: phrase lookup still wins (no letters from phrase words)
{
  const r = englishToGloss("nice to meet you");
  check("G8 phrase beats per-letter fallback",
    r.glosses.length === 1 && r.glosses[0].gloss === "NICE TO MEET YOU",
    JSON.stringify(r.glosses));
}

// G9: every letter sign exists in the real model vocabulary
{
  const manifestUrl = new URL("../public/avatar-data/manifest.json", import.meta.url);
  const manifest = JSON.parse(fs.readFileSync(manifestUrl, "utf-8"));
  const labels: string[] = Array.isArray(manifest) ? manifest : manifest.labels;
  const labelSet = new Set(labels.map((l) => String(l).toUpperCase()));
  const missing = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").filter((l) => !labelSet.has(l));
  check("G9 all 26 letter signs in model vocab", missing.length === 0,
    `missing: ${missing.join(",")}`);
}

// G10: empty input
{
  const r = englishToGloss("   ");
  check("G10 empty input yields no glosses", r.glosses.length === 0,
    JSON.stringify(r.glosses));
}

console.log(`\n${checks} checks, ${failures} failures`);
if (failures > 0) process.exit(1);
