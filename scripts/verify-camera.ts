/**
 * CAMERA GATE
 * ---------------------------------------------------------------------------
 * End-to-end check that the webcam feed actually reaches the <video>
 * element on the Translate and Practice views, run against a test-mode
 * build with Firebase stubbed (firebase-test-backend.ts) and a synthetic
 * camera device (Chrome's --use-fake-device-for-media-stream flags).
 *
 * Regression this gate exists for: useCamera.start() assigned the stream
 * to videoRef.current only when the element already existed, but
 * CameraPreview mounts the <video> only AFTER status === "ready" — so on
 * a cold start the stream was acquired (permission granted, camera light
 * on) yet never attached, leaving a blank white panel. This gate asserts
 * the stream is attached, frames are flowing, and playback advances.
 */
import { launch, type Page } from "puppeteer-core";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import {
  authStubLog,
  clickByText,
  installAuthBackend,
  signInViaUi,
  type AuthStubLog,
} from "./firebase-test-backend";

const ROOT = resolve(process.cwd());
const DIST = join(ROOT, "dist-verify");
const PORT = 4321;

const MIME: Record<string, string> = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".glb": "model/gltf-binary", ".wasm": "application/wasm",
};

const BROWSER_CANDIDATES = [
  process.env.CHROME_PATH,
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
].filter((p): p is string => !!p);

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The test bundle is rebuilt every run: a stale dist-verify would give
 *  false confidence exactly in the attach-path regression this gate guards. */
async function buildTestBundle() {
  console.log("Building test bundle (Vite mode=test, loads .env.test)…");
  const built = spawnSync(
    process.platform === "win32" ? "npx.cmd" : "npx",
    ["vite", "build", "--mode", "test", "--outDir", "dist-verify"],
    { cwd: ROOT, stdio: "inherit", shell: process.platform === "win32" }
  );
  if (built.status !== 0) {
    console.error("Test-mode build failed.");
    process.exit(1);
  }
}

async function serve() {
  const server = createServer(async (req, res) => {
    const url = (req.url ?? "/").split("?")[0];
    let p = join(DIST, url === "/" ? "index.html" : decodeURIComponent(url));
    if (!existsSync(p)) p = join(DIST, "index.html");
    try {
      const body = await readFile(p);
      res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  await new Promise<void>((r) => server.listen(PORT, r));
  return server;
}

interface CameraProbe {
  exists: boolean;
  attached: boolean;
  readyState: number;
  videoWidth: number;
  currentTime: number;
}

async function probeCamera(page: Page): Promise<CameraProbe> {
  return page.evaluate(() => {
    const v = document.querySelector("video");
    return {
      exists: !!v,
      attached: !!v && !!v.srcObject,
      readyState: v ? v.readyState : 0,
      videoWidth: v ? v.videoWidth : 0,
      currentTime: v ? v.currentTime : 0,
    };
  });
}

/** True once a <video> exists with an attached stream and decoded frames. */
async function feedIsLive(page: Page, timeoutMs = 30_000): Promise<boolean> {
  try {
    await page.waitForFunction(
      () => {
        const v = document.querySelector("video");
        return (
          !!v &&
          !!v.srcObject &&
          v.readyState >= 2 &&
          v.videoWidth > 0
        );
      },
      { timeout: timeoutMs }
    );
    return true;
  } catch {
    return false;
  }
}

async function main() {
  await buildTestBundle();
  const server = await serve();

  const exe = BROWSER_CANDIDATES.find((p) => existsSync(p));
  if (!exe) {
    console.error("No Chromium-based browser found for the camera gate.");
    server.close();
    process.exit(1);
  }

  const browser = await launch({
    executablePath: exe,
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-gpu",
      "--use-gl=swiftshader",
      // Synthetic camera: getUserMedia returns a test-pattern stream and
      // the permission prompt is auto-accepted.
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
    ],
  });

  const authLog: AuthStubLog = authStubLog();

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 414, height: 900, deviceScaleFactor: 1 });
    await installAuthBackend(page, authLog);

    await page.goto(`http://localhost:${PORT}/`, {
      waitUntil: "domcontentloaded",
      timeout: 60_000,
    });
    await signInViaUi(page);
    check(
      "sign-in reaches the app",
      await page
        .waitForFunction(
          () => /START TRANSLATING/i.test(document.body.innerText),
          { timeout: 30_000 }
        )
        .then(() => true)
        .catch(() => false)
    );

    /* Translate view — cold camera start -------------------------------- */
    console.log("\nTranslate view:");
    check("navigated to Translate", await clickByText(page, /Translate/i));
    check("live feed renders (stream attached, frames flowing)", await feedIsLive(page));
    const probe = await probeCamera(page);
    check("camera stream attached to <video>", probe.attached, JSON.stringify(probe));
    check("video is decoding (videoWidth > 0)", probe.videoWidth > 0, `videoWidth=${probe.videoWidth}`);
    const bodyText = await page.evaluate(() => document.body.innerText);
    check(
      "no camera error panel",
      !/Camera permission was denied|No camera device|already in use by another/i.test(bodyText)
    );

    /* Practice view — remount path -------------------------------------- */
    console.log("\nPractice view:");
    check("navigated to Practice", await clickByText(page, /Practice/i));
    check("feed re-attached after view remount", await feedIsLive(page));
    const probe2 = await probeCamera(page);
    check("stream attached on remounted <video>", probe2.attached, JSON.stringify(probe2));
    check("practice feed decodes frames", probe2.videoWidth > 0, `videoWidth=${probe2.videoWidth}`);

    /* Playback actually advances, not just a frozen first frame -------- */
    const before = probe2.currentTime;
    await sleep(2500);
    const probe3 = await probeCamera(page);
    check(
      "playback advances (currentTime moves)",
      probe3.currentTime > before,
      `${before.toFixed(2)}s -> ${probe3.currentTime.toFixed(2)}s`
    );
  } finally {
    await browser.close();
    server.close();
  }

  if (failures) {
    console.error(`\nCAMERA VERIFY FAILED: ${failures} check(s)`);
    process.exit(1);
  }
  console.log("\nCAMERA VERIFY PASSED");
}

void main();
