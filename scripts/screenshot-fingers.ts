/**
 * Driver: bundles scripts/screenshot-fingers.client.ts with esbuild, serves
 * the real Avery.glb plus recordings from public/avatar-data, renders them
 * in headless Edge, and captures a PNG grid for visual inspection of the
 * avatar's finger articulation.
 *
 * Run with: npx tsx scripts/screenshot-fingers.ts [out.png]
 */
import * as fs from "fs";
import * as http from "http";
import * as path from "path";
import { fileURLToPath } from "url";
import { build } from "esbuild";
import puppeteer, { type Browser } from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const GLB = path.join(ROOT, "public", "models", "Avery.glb");
const DATA_DIR = path.join(ROOT, "public", "avatar-data");
const CLIENT = path.join(__dirname, "screenshot-fingers.client.ts");
const OUT = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(__dirname, "screenshot-fingers.png");

const BROWSER_CANDIDATES = [
    process.env.CHROME_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
].filter((p): p is string => !!p);

function findBrowser(): string {
    const found = BROWSER_CANDIDATES.find((p) => fs.existsSync(p));
    if (!found) throw new Error("No Chrome/Edge executable found. Set CHROME_PATH.");
    return found;
}

const PAGE_HTML = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Finger articulation screenshot</title>
    <style>
      html,body{margin:0;background:#101014;font-family:system-ui,sans-serif}
      #grid{display:grid;grid-template-columns:repeat(4,auto);gap:10px;padding:10px;justify-content:center}
      .cell{position:relative}
      canvas{display:block;background:#101014}
      .cap{color:#e8e8ee;font-size:14px;text-align:center;padding:4px 0;background:#101014}
    </style>
  </head>
  <body>
    <div id="grid"></div>
    <script src="/bundle.js"></script>
  </body>
</html>`;

async function main() {
  console.log("bundling client with esbuild ...");
  const built = await build({
    entryPoints: [CLIENT],
    bundle: true,
    write: false,
    format: "iife",
    platform: "browser",
    target: "es2020",
    logLevel: "error",
  });
  const bundle = built.outputFiles[0].text;

  const glb = fs.readFileSync(GLB);
  const server = http.createServer((req, res) => {
    const url = (req.url ?? "/").split("?")[0];
    if (url === "/" || url === "/index.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(PAGE_HTML);
    } else if (url === "/bundle.js") {
      res.writeHead(200, { "content-type": "text/javascript; charset=utf-8" });
      res.end(bundle);
    } else if (url.startsWith("/data/")) {
      const name = path.basename(url);
      const file = path.join(DATA_DIR, name);
      if (!fs.existsSync(file)) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(fs.readFileSync(file));
    } else if (url === "/Avery.glb") {
      res.writeHead(200, { "content-type": "model/gltf-binary" });
      res.end(glb);
    } else if (url === "/favicon.ico") {
      res.writeHead(204);
      res.end();
    } else {
      res.writeHead(404);
      res.end("not found");
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  const port = typeof addr === "object" && addr ? addr.port : 0;

  const executablePath = findBrowser();
  console.log(`browser : ${executablePath}`);
  console.log(`server  : http://127.0.0.1:${port}/`);

  let browser: Browser | undefined;
  const pageErrors: string[] = [];
  try {
    browser = await puppeteer.launch({
      executablePath,
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--window-size=1520,900",
      ],
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1520, height: 900 });
    page.on("pageerror", (e) => pageErrors.push(String(e)));
    page.on("console", (m) => {
      if (m.type() === "error") pageErrors.push(`console: ${m.text()}`);
    });

    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 60000 });
    await page.waitForFunction("window.__READY__ !== undefined", { timeout: 180000 });
    const err = await page.evaluate("window.__ERROR__ ?? null");
    if (err) {
      console.error("in-page error:\n" + err);
      process.exitCode = 1;
      return;
    }

    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    await page.screenshot({ path: OUT, fullPage: true });
    console.log(`wrote ${OUT}`);
    if (pageErrors.length) {
      console.log("page errors:");
      for (const e of pageErrors) console.log("  " + e);
    } else {
      console.log("no page errors");
    }
  } finally {
    await browser?.close();
    server.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
