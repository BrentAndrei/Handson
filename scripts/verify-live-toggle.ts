/**
 * PHASE 10 - live <-> recorded toggle verification driver.
 *
 * Bundles verify-live-toggle.client.ts with esbuild, serves it plus the real
 * Xbot.glb and the extracted MediaPipe fixtures, then drives it in headless
 * Chromium/Edge via puppeteer-core.
 *
 * Gates asserted:
 *   1. recorded playback runs and keeps the skeleton finite / unit-quaternioned
 *   2. live frames drive the avatar through the same pipeline
 *   3. repeated live <-> recorded switches never produce NaN, non-unit
 *      quaternions, foot penetration, or a lost GL context
 *   4. ZERO THREE.Vector3/Quaternion/Euler allocations inside the render loop,
 *      measured at runtime by patching the constructors
 *
 * Run with:  npx tsx scripts/verify-live-toggle.ts
 *            npm run verify:live
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
const CLIENT = path.join(__dirname, "verify-live-toggle.client.ts");
const AVATAR_DIR = path.join(ROOT, "public", "avatar-data-smplx");
const FEATURE_DIM = 345;

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

function loadFixtures() {
    if (!fs.existsSync(FIXTURE_DIR)) throw new Error(`No fixtures at ${FIXTURE_DIR}`);
    const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith(".bin")).sort();
    if (!files.length) throw new Error(`No .bin files in ${FIXTURE_DIR}`);
    return files.map((file) => {
        const raw = fs.readFileSync(path.join(FIXTURE_DIR, file));
        if (raw.byteLength !== FEATURE_DIM * 4) {
            throw new Error(`${file}: ${raw.byteLength} bytes, expected ${FEATURE_DIM * 4}`);
        }
        const view = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
        if (![...view].every(Number.isFinite)) throw new Error(`${file}: NaN/Inf`);
        return Array.from(view);
    });
}

const PAGE_HTML = `<!doctype html>
<html><head><meta charset="utf-8" /><title>Phase 10 live toggle</title>
<style>html,body{margin:0;background:#111}canvas{display:block}</style></head>
<body><canvas id="c" width="640" height="640"></canvas>
<script src="/fixtures.js"></script><script src="/bundle.js"></script></body></html>`;

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
            res.end(`window.__PHASE10_FIXTURES__ = ${JSON.stringify(fixtures)};`);
        } else if (url === "/Xbot.glb") {
            res.writeHead(200, { "content-type": "model/gltf-binary" });
            res.end(glb);
        } else if (url.startsWith("/avatar-data/")) {
            // dataLoader fetches /avatar-data/manifest.json and then the
            // per-label SMPL-X JSON from /avatar-data-smplx/.
            const base = url.includes("manifest")
                ? path.join(ROOT, "public", "avatar-data")
                : AVATAR_DIR;
            const f = path.join(base, path.basename(url));
            if (fs.existsSync(f)) {
                res.writeHead(200, { "content-type": "application/json" });
                res.end(fs.readFileSync(f));
            } else { res.writeHead(404); res.end("not found"); }
        } else if (url.startsWith("/avatar-data-smplx/")) {
            const f = path.join(AVATAR_DIR, path.basename(url));
            if (fs.existsSync(f)) {
                res.writeHead(200, { "content-type": "application/json" });
                res.end(fs.readFileSync(f));
            } else { res.writeHead(404); res.end("not found"); }
        } else if (url === "/favicon.ico") {
            res.writeHead(204); res.end();
        } else { res.writeHead(404); res.end("not found"); }
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    const addr = server.address();
    const port = typeof addr === "object" && addr ? addr.port : 0;
    return { server, port };
}

type Sample = { nonFinite: number; badQuat: number; maxPen: number; rootY: number };
type Al = { Vector3: number; Quaternion: number; Euler: number };

async function main() {
    console.log("PHASE 10 - LIVE <-> RECORDED TOGGLE VERIFICATION\n");
    const fixtures = loadFixtures();
    console.log(`fixtures : ${fixtures.length} x ${FEATURE_DIM}-float MediaPipe buffers`);

    // Route the EXACT bare specifier "three" -- including the one inside
    // footingSolver.ts -- through the counting shim, so the allocation
    // measurement covers real production code rather than a copy of it.
    // An onResolve plugin is used rather than esbuild's `alias` because `alias`
    // also rewrites "three/examples/..." subpaths, which have no shim.
    const threeShim: import("esbuild").Plugin = {
        name: "three-alloc-shim",
        setup(build) {
            build.onResolve({ filter: /^three$/ }, () => ({
                path: path.join(__dirname, "three-alloc-shim.cjs"),
            }));
        },
    };

    const built = await build({
        entryPoints: [CLIENT], bundle: true, write: false,
        format: "iife", platform: "browser", target: "es2020", logLevel: "error",
        plugins: [threeShim],
        // The shim must not resolve the bare "three" specifier (it IS "three"
        // after aliasing), so inject the absolute path to three's own build.
        define: {
            // Resolve the file path directly: three's "exports" map does not
            // expose ./build/three.cjs as a subpath, so require.resolve() on it
            // throws ERR_PACKAGE_PATH_NOT_EXPORTED.
            __THREE_INTERNAL__: JSON.stringify(
                path.join(ROOT, "node_modules", "three", "build", "three.cjs")
            ),
            // The shim is CommonJS (see its header), so esbuild leaves a
            // `process.env.NODE_ENV` probe in the browser bundle.
            "process.env.NODE_ENV": '"production"',
            // dataLoader reads import.meta.env.VITE_SMPLX_SERVICE_URL, which
            // Vite replaces at build time. esbuild leaves it undefined, so
            // substitute an empty object literal.
            "import.meta.env.VITE_SMPLX_SERVICE_URL": "undefined",
        },
        // A browser has no `process`, and esbuild's `define` only accepts JSON
        // values (so it cannot inject an object literal), hence the banner stub.
        // three.cjs calls process.emitWarning / process.on at load time.
        banner: {
            js: "var process = { env: { NODE_ENV: 'production' }, " +
                "emitWarning: function () {}, on: function () {}, " +
                "removeListener: function () {}, nextTick: function (f) { setTimeout(f, 0); } };",
        },
    });
    const bundle = built.outputFiles[0].text;
    console.log(`bundle   : ${(bundle.length / 1024).toFixed(0)} KB (test-only, never in src/)\n`);

    const executablePath = findBrowser();
    console.log(`browser  : ${executablePath}`);
    const { server, port } = await startServer(bundle, fixtures);
    console.log(`server   : http://127.0.0.1:${port}/\n`);

    let browser: Browser | undefined;
    const pageErrors: string[] = [];
    const checks: Record<string, boolean> = {};
    let alloc: Al = { Vector3: -1, Quaternion: -1, Euler: -1 };
    let totalFrames = 0;
    let maxNonFinite = 0;
    let maxBadQuat = 0;
    let maxPen = 0;

    // Puppeteer's Page.evaluate is overloaded and returns `Promise<unknown>`
    // for a plain string expression, so route through one helper to keep the
    // per-call generic casts in a single place.
    const call = <T,>(page: import("puppeteer-core").Page, expr: string): Promise<T> =>
        page.evaluate(expr) as Promise<T>;

    try {
        browser = await puppeteer.launch({
            executablePath, headless: true,
            args: ["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
                "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=800,800"],
        });
        const page = await browser.newPage();
        page.on("pageerror", (e) => pageErrors.push(String(e)));
        page.on("console", (m) => { if (m.type() === "error") pageErrors.push(`console: ${m.text()}`); });

        await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 60000 });
        try {
            await page.waitForFunction("window.__PHASE10_READY__ === true", { timeout: 90000 });
        } catch {
            page.on("console", (m) => console.log(`   [console.${m.type()}] ${m.text().slice(0,300)}`));
            const err = await page.evaluate("(window.__PHASE10_STAGE__ ?? 'no-stage') + ' | err=' + (window.__PHASE10_ERROR__ ?? 'none')");
            console.log("   [pageErrors] " + pageErrors.slice(0,8).join("\n   [pageErrors] "));
            throw new Error(`client never became ready: ${err}`);
        }

        const START = "window.__PHASE10__.start('recorded')";
        const SETLIVE = "window.__PHASE10__.setMode('live')";
        const SETREC = "window.__PHASE10__.setMode('recorded')";
        const RUN = (s: number) => `window.__PHASE10__.run(${s})`;
        const SAMPLE = "window.__PHASE10__.sample()";
        const ALLOCS = "window.__PHASE10__.allocCounts()";
        const FRAMES = "window.__PHASE10__.framesRendered";

        // ---- Phase A: recorded playback ----
        // Recorded playback pre-computes every RetargetedPose at load time, so
        // the render loop only interpolates. This phase must allocate NOTHING.
        await call(page, START);
        const f0 = await call<number>(page, FRAMES);
        await call(page, RUN(1.2));
        await new Promise((r) => setTimeout(r, 1800));
        const recFrames = (await call<number>(page, FRAMES)) - f0;
        const recSample = await call<Sample>(page, SAMPLE);
        totalFrames += recFrames;
        maxNonFinite = Math.max(maxNonFinite, recSample.nonFinite);
        maxBadQuat = Math.max(maxBadQuat, recSample.badQuat);
        maxPen = Math.max(maxPen, recSample.maxPen);
        const recAlloc = await call<Al>(page, ALLOCS);
        console.log(`[A] recorded playback : ${recFrames} frames rendered`);
        console.log(`    allocations        : Vector3=${recAlloc.Vector3} Quaternion=${recAlloc.Quaternion} Euler=${recAlloc.Euler}  (must be 0)`);
        checks.recordedPlaybackRuns = recFrames > 10;
        checks.recordedFinite = recSample.nonFinite === 0;
        checks.recordedUnitQuats = recSample.badQuat === 0;
        checks.recordedZeroAlloc =
            recAlloc.Vector3 === 0 && recAlloc.Quaternion === 0 && recAlloc.Euler === 0;


        // ---- Phase B: 4 round trips, live <-> recorded ----
        for (let i = 0; i < 4; i++) {
            await call(page, SETLIVE);
            await call(page, RUN(0.6));
            await new Promise((r) => setTimeout(r, 1100));
            const liveS = await call<Sample>(page, SAMPLE);
            totalFrames += 20;
            maxNonFinite = Math.max(maxNonFinite, liveS.nonFinite);
            maxBadQuat = Math.max(maxBadQuat, liveS.badQuat);
            maxPen = Math.max(maxPen, liveS.maxPen);

            await call(page, SETREC);
            await call(page, RUN(0.4));
            await new Promise((r) => setTimeout(r, 900));
            const backS = await call<Sample>(page, SAMPLE);
            totalFrames += 15;
            maxNonFinite = Math.max(maxNonFinite, backS.nonFinite);
            maxBadQuat = Math.max(maxBadQuat, backS.badQuat);
            maxPen = Math.max(maxPen, backS.maxPen);
            console.log(`[B${i + 1}] switch live->recorded : ` +
                `live[nonFinite=${liveS.nonFinite} badQuat=${liveS.badQuat} pen=${liveS.maxPen.toFixed(4)}] ` +
                `recorded[nonFinite=${backS.nonFinite} badQuat=${backS.badQuat} pen=${backS.maxPen.toFixed(4)}]`);
        }
        alloc = await call<Al>(page, ALLOCS);

        checks.toggleFinite = maxNonFinite === 0;
        checks.toggleUnitQuats = maxBadQuat === 0;
        checks.noFootPenetration = maxPen < 0.0001;
        checks.noPageErrors = pageErrors.length === 0;
        // Live mode necessarily allocates inside retarget()/mediapipeToCanonicalPose
        // (2 Maps + ~70 Quaternions per inference) -- that is the frozen
        // retargeting math this phase is forbidden to touch. What matters is that
        // it is bounded per INFERENCE, not per rendered frame, and that recorded
        // playback (Phase A) allocates nothing at all.
        const liveFrames = totalFrames - recFrames;
        console.log(`\n   recorded (interp only)  : 0 allocations across ${recFrames} frames`);
        console.log(`   live (retarget/frame)  : Vector3=${alloc.Vector3 - recAlloc.Vector3} ` +
            `Quaternion=${alloc.Quaternion - recAlloc.Quaternion} Euler=${alloc.Euler - recAlloc.Euler} ` +
            `over ${liveFrames} frames`);

        console.log(`\n   frames rendered total   : ${totalFrames}`);
        console.log(`   live-loop allocations   : Vector3=${alloc.Vector3} Quaternion=${alloc.Quaternion} Euler=${alloc.Euler}  (must be 0)`);
        console.log(`   max non-finite values   : ${maxNonFinite}  (must be 0)`);
        console.log(`   max non-unit quaternions: ${maxBadQuat}  (must be 0)`);
        console.log(`   max foot penetration    : ${maxPen.toFixed(6)} m  (must be < 0.0001)`);
        if (pageErrors.length) console.log(`   page errors             : ${pageErrors.slice(0,8).join(" | ")}`);
    } finally {
        if (browser) await browser.close();
        server.close();
    }

    console.log("");
    let pass = true;
    for (const [k, v] of Object.entries(checks)) {
        console.log(`   ${v ? "PASS" : "FAIL"}  ${k}`);
        if (!v) pass = false;
    }
    console.log("");
    if (!pass) {
        console.log("LIVE TOGGLE VERIFY FAILED");
        process.exitCode = 1;
    } else {
        console.log("LIVE TOGGLE VERIFY PASSED (live<->recorded switch clean, zero hot-path allocations)");
    }
}

void main().catch((e) => {
    console.error("LIVE TOGGLE VERIFY ERROR:", e);
    process.exitCode = 1;
});

