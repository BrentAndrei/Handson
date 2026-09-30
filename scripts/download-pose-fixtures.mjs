/**
 * PHASE 4 tooling — download real pose-photo candidates for body validation.
 *
 * Queries the Wikimedia Commons search API for each required pose class and
 * downloads the top image candidates into scripts/fixtures/poses/candidates/.
 * A candidates.json manifest records title + description URL for attribution.
 *
 * MIME/asset sanity (hard gate for Phase 4):
 *   - the API must report image/jpeg or image/png for the file, AND
 *   - the downloaded bytes must actually start with a JPEG (FF D8 FF) or a
 *     PNG (89 50 4E 47) magic number.
 * Anything else (PDF, SVG, TIFF, WebP, an HTML error page, ...) is rejected, so
 * the candidates directory can never contain a non-image asset.
 *
 * Rate limiting: one request at a time, a fixed politeness delay between
 * downloads, and 429/503 backoff (honouring Retry-After) so we never hammer
 * the Commons API.
 *
 * Run with:  node scripts/download-pose-fixtures.mjs
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "fixtures", "poses", "candidates");
mkdirSync(OUT_DIR, { recursive: true });

const UA = "HandsonPhase4Fixtures/1.0 (offline test fixture download; contact: local dev)";

/** Hard cap per pose class. */
const PER_CLASS = 3;

/** The ONLY two MIME types Phase 4 accepts. Everything else is discarded. */
const ALLOWED_MIME = new Set(["image/jpeg", "image/png"]);

/** pose-class search terms — camera-facing, single-person framing preferred.
 *  The first entry is the spec's sanitized query; the rest are sanitized
 *  fallbacks tried in order only if the primary cannot yield PER_CLASS files. */
const QUERIES = {
  tpose: [
    "person standing T-pose exercise photo",
    "man arms outstretched standing photograph",
    "woman arms extended sideways standing photo",
  ],
  neutral: [
    "standing person front view full length photo",
    "person standing facing camera full body photo",
    "standing person arms at sides photograph",
  ],
  arms_both: [
    "jumping jacks exercise fitness photo",
    "jumping jack exercise photograph",
    "star jump exercise person photo",
  ],
  arm_left_raised: [
    "person raising left arm photo",
    "man one arm raised overhead photo",
    "woman arm raised up photograph",
  ],
  arm_right_raised: [
    "person waving right hand photo",
    "man waving hand at camera photo",
    "woman waving hello photograph",
  ],
  elbow_bent_left: [
    "dumbbell bicep curl exercise photo",
    "biceps curl with dumbbell photograph",
    "arm curl exercise gym photo",
  ],
  elbow_bent_right: [
    "one arm bicep curl workout photo",
    "bicep curl dumbbell exercise photo",
    "single arm dumbbell curl photograph",
    "curling exercise arm gym photo",
  ],
  torso_lean: [
    "person side stretch exercise photo",
    "standing side bend stretch photograph",
    "torso side bend exercise photo",
  ],
  asymmetric: [
    "standing athlete portrait photo",
    "athlete standing pose photograph",
    "person standing contrapposto photo",
  ],
};

const API = "https://commons.wikimedia.org/w/api.php";
const POLITE_DELAY_MS = 700;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Magic-number check — the byte-level guarantee behind the MIME gate. */
function sniffImage(buf) {
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return "jpg";
  if (
    buf.length >= 8 &&
    buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47 &&
    buf[4] === 0x0d && buf[5] === 0x0a && buf[6] === 0x1a && buf[7] === 0x0a
  ) {
    return "png";
  }
  return null;
}

async function fetchRetry(url, tries = 4) {
  let lastErr = "unknown";
  for (let a = 0; a < tries; a++) {
    let res;
    try {
      res = await fetch(url, { headers: { "User-Agent": UA } });
    } catch (e) {
      lastErr = e.message ?? String(e);
      await sleep(2000 * (a + 1));
      continue;
    }
    if (res.status === 429 || res.status === 503) {
      const retryAfter = Number(res.headers.get("retry-after"));
      const wait =
        Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 4000 * (a + 1);
      console.log(`    HTTP ${res.status} — backing off ${wait}ms`);
      await sleep(wait);
      lastErr = `HTTP ${res.status}`;
      continue;
    }
    return res;
  }
  throw new Error(lastErr);
}

async function search(term) {
  const url =
    `${API}?action=query&generator=search` +
    `&gsrsearch=${encodeURIComponent(term)}&gsrnamespace=6&gsrlimit=20` +
    `&prop=imageinfo&iiprop=url%7Cmime&iiurlwidth=800&format=json`;
  const res = await fetchRetry(url);
  if (!res.ok) throw new Error(`search "${term}": HTTP ${res.status}`);
  const json = await res.json();
  const pages = Object.values(json.query?.pages ?? {});

  let skippedMime = 0;
  const hits = [];
  for (const p of pages) {
    const info = p.imageinfo?.[0];
    if (!info) continue;
    // ---- MIME gate: image/jpeg and image/png only ------------------------
    if (!ALLOWED_MIME.has(info.mime)) {
      skippedMime++;
      continue;
    }
    const assetUrl = info.thumburl ?? info.url;
    if (!assetUrl) continue;
    hits.push({
      title: p.title,
      mime: info.mime,
      url: assetUrl,
      descriptionurl: info.descriptionurl,
      index: p.index ?? 99,
    });
  }
  hits.sort((a, b) => a.index - b.index);
  return { hits, skippedMime };
}

const manifest = [];
const rejected = [];
let rejectedMime = 0;
let rejectedBytes = 0;
let rejectedDupe = 0;
/** A Commons file is only ever used once, across ALL pose classes — otherwise a
 *  fallback query that resolves to the same result set as another class would
 *  fill both with byte-identical images and the "n" sample would be correlated. */
const seenTitles = new Set();

console.log(`PHASE 4 fixture download -> ${OUT_DIR}`);
console.log(`accepting MIME: ${[...ALLOWED_MIME].join(", ")} only; max ${PER_CLASS} per class\n`);

for (const [poseClass, terms] of Object.entries(QUERIES)) {
  let i = 0;
  let usedQuery = null;
  for (const term of terms) {
    if (i >= PER_CLASS) break;
    try {
      const { hits, skippedMime } = await search(term);
      rejectedMime += skippedMime;
      const star = usedQuery ? " *" : " ";
      usedQuery = usedQuery ? `${usedQuery} + fallback` : term;
      console.log(
        `${star}[${poseClass}] "${term}" -> ${hits.length} MIME-ok hits` +
          (skippedMime ? `, ${skippedMime} rejected by MIME gate` : "")
      );

      for (const r of hits) {
        if (i >= PER_CLASS) break;

        // ---- cross-class duplicate gate ---------------------------------
        if (seenTitles.has(r.title)) {
          rejectedDupe++;
          rejected.push({ poseClass, title: r.title, apiMime: r.mime, reason: "already used by another pose class" });
          continue;
        }

        const res = await fetchRetry(r.url);
        if (!res.ok) {
          console.log(`  [${poseClass}] HTTP ${res.status} on candidate — skipped`);
          continue;
        }
        await sleep(POLITE_DELAY_MS); // stay under Wikimedia rate limits
        const buf = Buffer.from(await res.arrayBuffer());

        // ---- byte-level MIME gate (the real guarantee) ----------------------
        const sniffed = sniffImage(buf);
        if (!sniffed) {
          rejectedBytes++;
          rejected.push({ poseClass, title: r.title, apiMime: r.mime, reason: "magic bytes are not JPEG/PNG" });
          console.log(`  [${poseClass}] REJECTED (not JPEG/PNG): ${r.title}`);
          continue;
        }
        if (buf.length < 15000) {
          rejectedBytes++;
          rejected.push({ poseClass, title: r.title, apiMime: r.mime, reason: `thumbnail stub (${buf.length}B)` });
          console.log(`  [${poseClass}] REJECTED (thumbnail stub, ${buf.length}B): ${r.title}`);
          continue;
        }

        const file = `${poseClass}_${i}.${sniffed}`;
        writeFileSync(join(OUT_DIR, file), buf);
        seenTitles.add(r.title);
        manifest.push({
          poseClass,
          file,
          query: term,
          title: r.title,
          apiMime: r.mime,
          sniffedMime: `image/${sniffed === "jpg" ? "jpeg" : "png"}`,
          descriptionurl: r.descriptionurl,
          bytes: buf.length,
        });
        console.log(
          `  [${poseClass}] saved ${file.padEnd(26)} ${(buf.length / 1024).toFixed(0).padStart(5)} KB  ${r.title}`
        );
        i++;
      }
    } catch (e) {
      console.error(`  [${poseClass}] query "${term}" FAILED — ${e.message}`);
    }
  }
  if (i === 0) console.log(`  [${poseClass}] NO CANDIDATES from any of its ${terms.length} queries`);
}

writeFileSync(
  join(OUT_DIR, "candidates.json"),
  JSON.stringify(
    { generatedBy: "download-pose-fixtures.mjs", allowedMime: [...ALLOWED_MIME], accepted: manifest, rejected },
    null,
    2
  )
);

const perClass = {};
for (const m of manifest) perClass[m.poseClass] = (perClass[m.poseClass] ?? 0) + 1;
const missing = Object.keys(QUERIES).filter((k) => !perClass[k]);

console.log("\n============ PHASE 4 FIXTURE DOWNLOAD SUMMARY ============");
console.log(`accepted files     : ${manifest.length}`);
console.log(`by pose class      : ${Object.entries(perClass).map(([k, v]) => `${k}=${v}`).join(", ") || "(none)"}`);
console.log(`classes with none  : ${missing.join(", ") || "(none)"}`);
console.log(`file extensions    : ${[...new Set(manifest.map((m) => m.file.split(".").pop()))].join(", ") || "(none)"}`);
console.log(`non-image assets   : 0   (MIME gate + magic-byte gate)`);
console.log(`rejected           : ${rejectedMime} by API mime, ${rejectedBytes} by byte sniff, ${rejectedDupe} cross-class duplicate`);
console.log(`manifest           : ${join(OUT_DIR, "candidates.json")}`);

if (missing.length) {
  console.error(`\nFAIL: pose classes with no fixture: ${missing.join(", ")}`);
  process.exitCode = 1;
}
