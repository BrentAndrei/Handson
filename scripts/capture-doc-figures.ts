/**
 * Captures the six figures the documentation asks for, from the REAL build.
 * Serves ./dist over http (same approach as verify-ui, so no dev server and
 * no self-signed cert; screenshots match the shipped bundle).
 */
import { launch } from "puppeteer-core";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(process.cwd());
const DIST = join(ROOT, "dist");
const OUT = join(ROOT, "tmp", "doc");
const PORT = 4319;

const EDGE_CANDIDATES = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
];

const MIME: Record<string, string> = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
  ".wasm": "application/wasm", ".json": "application/json",
  ".glb": "model/gltf-binary", ".bin": "application/octet-stream",
};

const server = createServer(async (req, res) => {
  try {
    const url = (req.url ?? "/").split("?")[0];
    let p = join(DIST, decodeURIComponent(url === "/" ? "/index.html" : url));
    if (!existsSync(p)) p = join(DIST, "index.html");
    const buf = await readFile(p);
    res.writeHead(200, { "Content-Type": MIME[extname(p)] ?? "application/octet-stream" });
    res.end(buf);
  } catch {
    res.writeHead(404).end("not found");
  }
});

await new Promise<void>((r) => server.listen(PORT, r));
await mkdir(OUT, { recursive: true });

const exe = EDGE_CANDIDATES.find((p) => existsSync(p));
if (!exe) throw new Error("No Chromium browser found.");

const browser = await launch({
  executablePath: exe,
  headless: true,
  args: [
    "--no-sandbox", "--disable-gpu", "--use-gl=swiftshader",
    // Synthetic camera so Sign-to-Text renders a real video element
    // instead of hanging on the permission prompt.
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
  ],
});

await browser.defaultBrowserContext().overridePermissions(`http://localhost:${PORT}`, ["camera"]);

const page = await browser.newPage();
const errors: string[] = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.setViewport({ width: 1280, height: 860, deviceScaleFactor: 1 });
await page.goto(`http://localhost:${PORT}/`, { waitUntil: "networkidle0", timeout: 60_000 });
await new Promise((r) => setTimeout(r, 1500));

const shot = async (name: string) => {
  await page.screenshot({ path: join(OUT, name + ".png") });
  console.log("shot   " + name);
};

const clickText = async (t: string, p = page): Promise<boolean> => {
  const ok = await p.evaluate((txt) => {
    const els = Array.from(
      document.querySelectorAll<HTMLElement>("button, a, [role=button]"));
    const hit = els.find((e) =>
      (e.textContent ?? "").replace(/\s+/g, " ").toUpperCase().includes(txt.toUpperCase()));
    if (!hit) return false;
    hit.click();
    return true;
  }, t);
  await new Promise((r) => setTimeout(r, 1400));
  return ok;
};

// Types into the sentence box using the native setter, so React's onChange fires.
// Then dismisses the autocomplete dropdown (it overlays the gloss panel) and
// returns the rendered gloss so the capture can be verified, not assumed.
const setSentence = async (text: string, p = page): Promise<string | null> => {
  const ok = await p.evaluate((val) => {
    const box = document.querySelector<HTMLInputElement | HTMLTextAreaElement>(
      "input[type=text], textarea");
    if (!box) return false;
    const proto = box instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value")?.set?.call(box, val);
    box.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  }, text);
  if (!ok) return null;
  // The Enter handler lives on the textarea's onKeyDown, so the element must
  // genuinely hold focus. Setting .value alone does not focus it.
  await p.evaluate(() => {
    document.querySelector<HTMLTextAreaElement>("textarea, input[type=text]")?.focus();
  });
  await p.keyboard.press("Enter");
  await new Promise((r) => setTimeout(r, 3000));
  // Click a neutral heading to close the suggestion list, then read the gloss.
  return await p.evaluate(() => {
    document.querySelector("h1, h2, p")?.dispatchEvent(
      new MouseEvent("mousedown", { bubbles: true }));
    (document.activeElement as HTMLElement | null)?.blur();
    return document.body.innerText.replace(/\s+/g, " ").slice(0, 600);
  });
};

// Fig 1 - home screen. NOTE: the browser's own camera permission prompt is
// browser chrome and cannot be captured from inside the page; the in-app
// READY state is what this shows.
await shot("fig1-home");

// Fig 2 - Sign-to-Text with landmark overlay and captions.
await clickText("TRANSLATE");
await new Promise((r) => setTimeout(r, 1200));
await shot("fig2-sign-to-text");

// Fig 3 - Text-to-Sign lives in the AVATAR tab, not Translate (the sentence
// box is not present in the translate DOM - verified by probe).
const openAvatar = await clickText("AVATAR");
await new Promise((r) => setTimeout(r, 1200));
const gloss3 = await setSentence("Where is the library?");
await shot("fig3-text-to-sign");

// Fig 4 - negation sentence and its generated gloss.
const gloss4 = await setSentence("I do not like the weather.");
await shot("fig4-negation-gloss");

console.log("\nfig3 gloss panel: " + (gloss3 ?? "(no input found)"));
console.log("fig4 gloss panel: " + (gloss4 ?? "(no input found)"));

// Fig 5 - the DISPLAY / accessibility panel.
//
// HONESTY NOTE: the document's original "Performance monitoring (FPS, latency,
// memory)" feature does NOT exist in the shipped UI - grepping src/ for FPS,
// latency or memory readouts returns nothing. Rather than fabricate a HUD, this
// figure captures the runtime controls that ARE real: the status-bar tier
// indicator (visible on every screen) and the display panel that sets text size
// and contrast. Captured cropped to the right-hand column so it differs from
// fig 2 rather than duplicating it.
await page.evaluate(() => {
  document.querySelector('[aria-label*="ccessibility" i]')?.scrollIntoView({ block: "center" });
});
await new Promise((r) => setTimeout(r, 700));
await clickText("TRANSLATE");
await new Promise((r) => setTimeout(r, 1200));

// Crop to the right-hand column (status bar chips + display panel).
await page.screenshot({
  path: join(OUT, "fig5-performance.png"),
  clip: { x: 780, y: 0, width: 500, height: 700 },
});
console.log("shot   fig5-performance (cropped right column)");

// Fig 6 - camera-denied error state.
//
// This needs its OWN browser instance: the fake-device flags on the main one
// auto-accept the permission prompt, which silently defeats overridePermissions
// and produces a picture identical to fig 2. Launching without those flags and
// with camera explicitly blocked makes getUserMedia raise a real NotAllowedError,
// so the message shown is the app's genuine error path.
const deniedBrowser = await launch({
  executablePath: exe,
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--use-gl=swiftshader"],
});
const deniedCtx = await deniedBrowser.defaultBrowserContext();
await deniedCtx.overridePermissions(`http://localhost:${PORT}`, []);
const denied = await deniedBrowser.newPage();
await denied.setViewport({ width: 1280, height: 860, deviceScaleFactor: 1 });
await denied.goto(`http://localhost:${PORT}/`, { waitUntil: "networkidle0" });
await new Promise((r) => setTimeout(r, 1500));
await clickText("TRANSLATE", denied);
await new Promise((r) => setTimeout(r, 3000));
await denied.screenshot({ path: join(OUT, "fig6-camera-denied.png") });
const deniedText = await denied.evaluate(() => document.body.innerText.replace(/\s+/g, " "));
console.log("shot   fig6-camera-denied");
console.log("fig6 shows denial: " +
  /denied|not allowed|permission/i.test(deniedText) + "  ->  " +
  (deniedText.match(/[^ ]*permission[^ ]*/i)?.[0] ?? "no permission text"));
await deniedBrowser.close();

console.log("avatar nav worked: " + openAvatar);
console.log("page errors: " + errors.length +
  (errors.length ? " -> " + errors.slice(0, 3).join(" | ") : ""));

await browser.close();
server.close();
