/**
 * Render-check: drive the REAL pipeline in a real WebGL context with REAL
 * recorded clips (not synthetic fixtures), then screenshot the result.
 *
 * This is the only way to answer "does it look like a signer now?" -- the
 * numeric probes report bone positions, but the user sees pixels.
 *
 * Writes scripts/fixtures/render/clip_<label>_frame<i>.png
 *
 * Run: npx tsx scripts/render-check.ts
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as http from "http";
import { build } from "esbuild";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const GLB = path.join(ROOT, "public", "models", "Avery.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const OUT_DIR = path.join(__dirname, "fixtures", "render");
const CLIENT = path.join(__dirname, "render-check.client.ts");

const CLIPS = [
  { label: "ADOPT", frames: [0, 19, 39, 59, 78] },
  { label: "HELLO", frames: [0, 52, 104, 156, 207] },
  { label: "A", frames: [0, 23, 47, 70, 93] },
];

function findBrowser(): string {
  const cands = [
    process.env.CHROME_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  ].filter((p): p is string => !!p);
  const found = cands.find((p) => fs.existsSync(p));
  if (!found) throw new Error("No Chrome/Edge binary. Set CHROME_PATH.");
  return found;
}

function loadClip(label: string): number[][] {
  const p = path.join(DATA_DIR, `${label.replace(/[^a-zA-Z0-9]/g, "_")}.json`);
  const data = JSON.parse(fs.readFileSync(p, "utf8"));
  const frames = data.frames;
  const n = Math.floor(frames.length / 345);
  const out: number[][] = [];
  for (let i = 0; i < n; i++) out.push(Array.from(frames.slice(i * 345, (i + 1) * 345)));
  return out;
}

function serve(req: http.IncomingMessage, res: http.ServerResponse) {
  const u = new URL(req.url ?? "/", "http://localhost");
  if (u.pathname === "/bundle.js") {
    res.setHeader("Content-Type", "application/javascript; charset=utf-8");
    res.end(bundled);
  } else if (u.pathname === "/Avery.glb") {
    res.setHeader("Content-Type", "application/octet-stream");
    fs.createReadStream(GLB).pipe(res);
  } else if (u.pathname.startsWith("/avatar-data/")) {
    const file = path.join(ROOT, "public", u.pathname);
    if (fs.existsSync(file)) {
      res.setHeader("Content-Type", "application/json");
      fs.createReadStream(file).pipe(res);
    } else {
      res.statusCode = 404;
      res.end("not found");
    }
  } else {
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.end(html());
  }
}

function html(): string {
  return `<!doctype html><html><body>
<div id="stage" style="width:720px;height:720px"></div>
<script type="module" src="/bundle.js"></script>
</body></html>`;
}

let bundled: string;

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  bundled = await build({
    entryPoints: [CLIENT],
    bundle: true,
    format: "esm",
    platform: "browser",
    write: false,
    define: { "process.env.NODE_ENV": '"production"' },
  }).then((r) => r.outputFiles[0].text);

  const server = http.createServer(serve);
  await new Promise<void>((r) => server.listen(0, r));
  const port = (server.address() as any).port;
  const base = `http://localhost:${port}`;

  const browser = await puppeteer.launch({
    executablePath: findBrowser(),
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--use-gl=swiftshader", "--enable-webgl"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 720, height: 720, deviceScaleFactor: 1 });

  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto(base + "/", { waitUntil: "networkidle0" });

  const payload = JSON.stringify(CLIPS.map((c) => ({ label: c.label, frames: loadClip(c.label), samples: c.frames })));
  const result: any = await page.evaluate(async (payload: string) => {
    const data = JSON.parse(payload);
    const win = window as any;
    win.__CLIPS__ = data;
    win.__RUN__();
    await new Promise<void>((res) => { win.__DONE__ = res; });
    return { results: win.__RESULT__.results, screenshots: win.__SCREENSHOTS__ };
  }, payload);

  await browser.close();
  server.close();

  console.log(JSON.stringify(result.results, null, 2));
  if (errors.length) console.log("browser errors:", errors.slice(0, 10));

  const files: string[] = [];
  for (const [name, dataUrl] of Object.entries<any>(result.screenshots || {})) {
    const buf = Buffer.from(dataUrl.split(",", 2)[1], "base64");
    const p = path.join(OUT_DIR, name);
    fs.writeFileSync(p, buf);
    files.push(name);
  }
  files.sort();
  console.log(`\nscreenshots written to ${OUT_DIR}:`);
  for (const f of files) console.log("  " + f);
}

main().catch((e) => { console.error(e); process.exit(1); });
