/**
 * PHASE 4 — headless browser verification driver.
 *
 * Bundles scripts/verify-browser-pose.client.ts with esbuild, serves it plus
 * the real Xbot.glb and the extracted 345-float pose fixtures over a throwaway
 * loopback HTTP server, then drives it in headless Chromium/Edge through
 * puppeteer-core and asserts the browser-side gates:
 *
 *   1. zero NaN / Inf in any world matrix over 100 continuous rendered frames
 *   2. avatar center stays within +/-0.05 m of its bind origin
 *      (i.e. APPLY_ROOT_TRANSLATION = false is genuinely in force)
 *   3. the WebGL context survives, reports no GL errors, and the frames
 *      actually change bone rotations (not a frozen canvas)
 *
 * puppeteer-core is a devDependency only; nothing in src/ imports this file or
 * the client, so no test code reaches the production bundle.
 *
 * Run with:  npx tsx scripts/verify-browser-pose.ts
 *            npm run verify:browser
 */
import * as fs from "fs";
import * as http from "http";
import * as path from "path";
import { fileURLToPath } from "url";
import { build } from "esbuild";
import puppeteer, { type Browser } from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const GLB = path.join(ROOT, "public", "models", "Xbot.glb");
const FIXTURE_DIR = path.join(__dirname, "fixtures", "poses", "extracted");
const CLIENT = path.join(__dirname, "verify-browser-pose.client.ts");
const FEATURE_DIM = 345;

/** Candidate browsers, in preference order. */
const BROWSER_CANDIDATES = [
    process.env.CHROME_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
].filter((p): p is string => !!p);

function findBrowser(): string {
    const found = BROWSER_CANDIDATES.find((p) => fs.existsSync(p));
    if (!found) {
        throw new Error("No Chrome/Edge executable found. Set CHROME_PATH to a browser binary.");
    }
    return found;
}

function loadFixtures() {
    if (!fs.existsSync(FIXTURE_DIR)) {
        throw new Error(`No fixtures at ${FIXTURE_DIR} — run extract_landmarks.py first.`);
    }
    const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith(".bin")).sort();
    if (!files.length) throw new Error(`No .bin files in ${FIXTURE_DIR}`);

    return files.map((file) => {
        const raw = fs.readFileSync(path.join(FIXTURE_DIR, file));
        if (raw.byteLength !== FEATURE_DIM * 4) {
            throw new Error(`${file}: ${raw.byteLength} bytes, expected ${FEATURE_DIM * 4}`);
        }
        const view = new Float32Array(
            raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)
        );
        if (![...view].every(Number.isFinite)) {
            throw new Error(`${file}: landmark buffer contains NaN/Inf`);
        }
        return { name: file.replace(/\.bin$/, ""), data: Array.from(view) };
    });
}

const PAGE_HTML = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Phase 4 browser pose verification</title>
    <style>html,body{margin:0;background:#111}canvas{display:block}</style>
  </head>
  <body>
    <canvas id="c" width="640" height="640"></canvas>
    <script src="/fixtures.js"></script>
    <script src="/bundle.js"></script>
  </body>
</html>`;

async function startServer(bundle: string, fixtures: unknown) {
    const glb = fs.readFileSync(GLB);
    const server = http.createServer((req, res) => {
        const url = (req.url ?? "/").split("?")[0];
        if (url === "/" || url === "/index.html") {
            res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
            res.end(PAGE_HTML);
        } else if (url === "/bundle.js") {
            res.writeHead(200, { "content-type": "text/javascript; charset=utf-8" });
            res.end(bundle);
        } else if (url === "/fixtures.js") {
            res.writeHead(200, { "content-type": "text/javascript; charset=utf-8" });
            res.end(`window.__PHASE4_FIXTURES__ = ${JSON.stringify(fixtures)};`);
        } else if (url === "/Xbot.glb") {
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
    return { server, port };
}

type BrowserResult = {
    frames: number; renderedFrames: number; usedFixtures: number; uniqueFixtures: number;
    bones: number; maxRootDrift: number; maxCentroidDrift: number; totalMotion: number;
    nonFiniteCount: number; nonFinite: string[]; glErrors: number[]; contextLost: boolean;
    bindRoot: number[]; bindCentroid: number[];
    rootBBox: { xMin: number; xMax: number; yMin: number; yMax: number; zMin: number; zMax: number };
    rootSpan: { x: number; y: number; z: number };
    bindFloorY: number; minFootY: number; badQuatNorm: number;
    rendererInfo: Record<string, string>; checks: Record<string, boolean>; pass: boolean;
    error?: string;
};

async function main() {
    console.log("PHASE 4 — headless browser pose verification\n");

    const fixtures = loadFixtures();
    console.log(`fixtures : ${fixtures.length} x ${FEATURE_DIM}-float MediaPipe buffers`);

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
    console.log(`bundle   : ${(bundle.length / 1024).toFixed(0)} KB (test-only, never in src/)`);

    const executablePath = findBrowser();
    console.log(`browser  : ${executablePath}`);

    const { server, port } = await startServer(bundle, fixtures);
    console.log(`server   : http://127.0.0.1:${port}/\n`);

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
                "--window-size=800,800",
            ],
        });
        const page = await browser.newPage();
        page.on("pageerror", (e) => pageErrors.push(String(e)));
        page.on("console", (m) => {
            if (m.type() === "error") pageErrors.push(`console: ${m.text()}`);
        });

        await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 60000 });
        await page.waitForFunction("window.__PHASE4__ !== undefined", { timeout: 180000 });
        const r = (await page.evaluate("window.__PHASE4__")) as BrowserResult;

        if (r.error) {
            console.error("in-page error:\n" + r.error);
            process.exitCode = 1;
            return;
        }

        const f4 = (n: number) => n.toFixed(4);
        console.log("\n============ PHASE 4 BROWSER FRAME CHECK SUMMARY ============");
        console.log(`WebGL               : ${r.rendererInfo.renderer}`);
        console.log(`  vendor/version    : ${r.rendererInfo.vendor} / ${r.rendererInfo.version}`);
        console.log(`bones resolved      : ${r.bones}`);
        console.log(`frames requested    : ${r.frames}`);
        console.log(`frames rendered     : ${r.renderedFrames}`);
        console.log(`pose buffers applied: ${r.usedFixtures} (cycling ${r.uniqueFixtures} unique)`);
        console.log(`bind hips position  : [${r.bindRoot.map(f4).join(", ")}]`);
        console.log(`bind centroid       : [${r.bindCentroid.map(f4).join(", ")}]`);
        console.log(`max root drift      : ${f4(r.maxRootDrift)} m   (skeleton only, rotation retarget)`);
        console.log(`centroid excursion  : ${f4(r.maxCentroidDrift)} m   (pose-dependent, informational)`);
        console.log("");
        console.log("PHASE 7 ROOT TRANSLATION BOUNDING BOX (model.position, metres):");
        const bb = r.rootBBox;
        const g4 = (n: number) => n.toFixed(4);
        console.log(`   X : [${g4(bb.xMin)} .. ${g4(bb.xMax)}]   span ${g4(r.rootSpan.x)} m`);
        console.log(`   Y : [${g4(bb.yMin)} .. ${g4(bb.yMax)}]   span ${g4(r.rootSpan.y)} m`);
        console.log(`   Z : [${g4(bb.zMin)} .. ${g4(bb.zMax)}]   span ${g4(r.rootSpan.z)} m`);
        console.log(`   bind floor y=${g4(r.bindFloorY)}   lowest foot during run y=${g4(r.minFootY)}`);
        console.log(`   non-unit quaternions: ${r.badQuatNorm}`);
        console.log(`NaN/Inf occurrences : ${r.nonFiniteCount}${r.nonFinite.length ? " -> " + r.nonFinite.join("; ") : ""}`);
        console.log(`WebGL errors        : ${r.glErrors.length ? r.glErrors.join(", ") : "none"}`);
        console.log(`context lost        : ${r.contextLost}`);
        console.log(`bone motion (sum)   : ${f4(r.totalMotion)} rad (frames must actually update)`);
        console.log(`page errors         : ${pageErrors.length ? pageErrors.join(" | ") : "none"}`);
        console.log("\ngates:");
        for (const [k, v] of Object.entries(r.checks)) {
            console.log(`  [${v ? "ok  " : "FAIL"}] ${k}`);
        }

        const allPass = r.pass && pageErrors.length === 0;
        console.log(
            allPass
                ? `\nBROWSER VERIFY PASSED: ${r.renderedFrames} frames, 0 NaN/Inf, 0 GL errors, root drift ${f4(r.maxRootDrift)} m`
                : "\nBROWSER VERIFY FAILED"
        );
        if (!allPass) process.exitCode = 1;
    } finally {
        await browser?.close();
        server.close();
    }
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
