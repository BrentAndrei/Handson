/**
 * AUTHENTICATION GATE
 * ---------------------------------------------------------------------------
 * End-to-end checks for the Firebase login gate, run against a test-mode build
 * with Firebase's REST calls stubbed locally (see firebase-test-backend.ts).
 * The SDK is the real one — only the server is fake.
 *
 * Covers, in order:
 *   1. the gate blocks the app before authentication
 *   2. wrong credentials show a friendly message, not a raw SDK error
 *   3. a real sign-in reaches the app
 *   4. the session survives a reload (Firebase persistence, not localStorage)
 *   5. Account shows the real Firebase email and logs out
 *   6. after logout a reload shows the login screen again
 *   7. signup creates an account through Firebase
 *   8. forgot-password calls sendPasswordResetEmail and confirms it
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
  switchMode,
  TEST_EMAIL,
  TEST_PASSWORD,
  type AuthStubLog,
} from "./firebase-test-backend";

const ROOT = resolve(process.cwd());
const DIST = join(ROOT, "dist-verify");
const PORT = 4319;
const NEW_EMAIL = "newcomer@handson.test";

const MIME: Record<string, string> = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".glb": "model/gltf-binary", ".wasm": "application/wasm",
};

const EDGE_CANDIDATES = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
];

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
}

const text = (page: Page) => page.evaluate(() => document.body.innerText);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitForText(page: Page, re: RegExp, ms = 20_000): Promise<boolean> {
  try {
    await page.waitForFunction(
      (src) => new RegExp(src, "i").test(document.body.innerText),
      { timeout: ms },
      re.source
    );
    return true;
  } catch {
    return false;
  }
}

/** Clears the form fields so a retry does not append to the previous value. */
async function clearLoginForm(page: Page) {
  await page.evaluate(() => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, "value"
    )?.set;
    for (const el of Array.from(document.querySelectorAll("input"))) {
      setter?.call(el, "");
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
}

async function buildIfNeeded() {
  if (existsSync(join(DIST, "index.html"))) return;
  console.log("dist-verify missing — building test bundle…");
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

async function main() {
  await buildIfNeeded();
  const server = await serve();

  const exe = EDGE_CANDIDATES.find((p) => existsSync(p));
  if (!exe) {
    console.error("No Chromium-based browser found for the auth gate.");
    server.close();
    process.exit(1);
  }

  const browser = await launch({
    executablePath: exe,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu", "--use-gl=swiftshader"],
  });

  const authLog: AuthStubLog = authStubLog();

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 414, height: 900, deviceScaleFactor: 1 });
    await installAuthBackend(page, authLog, process.env.AUTH_STUB_DEBUG === "1");

    const go = () =>
      page.goto(`http://localhost:${PORT}/`, { waitUntil: "domcontentloaded", timeout: 60_000 });

    /* 1 — the gate ------------------------------------------------------- */
    console.log("Gate:");
    await go();
    await waitForText(page, /Log in/i);
    let body = await text(page);
    check("login screen shown before auth", /Log in/i.test(body));
    check("home screen is NOT rendered", !/START TRANSLATING/i.test(body));
    check("translator is NOT reachable", !/REAL-TIME/i.test(body));
    check("avatar is NOT reachable", !/OPEN 3D AVATAR/i.test(body));

    /* 2 — bad credentials ---------------------------------------------- */
    console.log("\nSign-in errors:");
    await page.waitForSelector('input[type="email"]');
    await page.type('input[type="email"]', TEST_EMAIL);
    await page.type('input[type="password"]', "wrong-password");
    await page.click('form button[type="submit"]');
    await waitForText(page, /Incorrect email or password/i);
    body = await text(page);
    check("wrong password shows a friendly message", /Incorrect email or password/i.test(body));
    check("raw Firebase error is not shown", !/auth\/|Firebase:|INVALID_LOGIN/i.test(body));

    /* 3 — real sign-in --------------------------------------------------- */
    console.log("\nSign-in:");
    await clearLoginForm(page);
    await page.type('input[type="email"]', TEST_EMAIL);
    await page.type('input[type="password"]', TEST_PASSWORD);
    await page.click('form button[type="submit"]');
    check("valid credentials reach the app", await waitForText(page, /START TRANSLATING/i));
    body = await text(page);
    check("app UI is fully rendered", /REAL-TIME/i.test(body) && /3D SIGN/i.test(body));

    /* 4 — session persistence ------------------------------------------- */
    console.log("\nSession:");
    await go();
    check("session survives a page reload", await waitForText(page, /START TRANSLATING/i));
    const stored = await page.evaluate(async () => ({
      // Firebase persists in IndexedDB; a HandSon-owned localStorage session
      // would mean auth had been reimplemented locally.
      handsonKeys: Object.keys(localStorage).filter((k) => /handson\.auth/i.test(k)),
      idb: indexedDB.databases ? (await indexedDB.databases()).map((d) => d.name) : [],
    }));
    check("no HandSon auth keys in localStorage", stored.handsonKeys.length === 0,
      JSON.stringify(stored.handsonKeys));
    check("Firebase persists the session in IndexedDB",
      stored.idb.some((n) => typeof n === "string" && n.includes("firebase")),
      JSON.stringify(stored.idb));

    /* 5 — account + logout ---------------------------------------------- */
    console.log("\nAccount:");
    check("navigated to Account", await clickByText(page, /Account/i));
    check("Account shows the real Firebase email",
      await waitForText(page, new RegExp(TEST_EMAIL.replace(/\./g, "\\."), "i")));
    body = await text(page);
    check("no placeholder account details", !/No account yet|Log in or sign up/i.test(body));

    check("logout button clicked", await clickByText(page, /log out/i));
    check("logout returns to the login screen", await waitForText(page, /Log in/i));
    check("app is gone after logout", !/START TRANSLATING/i.test(await text(page)));

    /* 6 — logout survives reload ---------------------------------------- */
    await go();
    check("reload after logout still shows login", await waitForText(page, /Log in/i, 15_000));
    check("no app access after logout+reload", !/START TRANSLATING/i.test(await text(page)));

    /* 7 — signup ---------------------------------------------------------- */
    console.log("\nSign-up:");
    check("switched to Create account", await switchMode(page, "signup"));
    await page.waitForSelector('input[type="email"]');
    let passwords = await page.$$('input[type="password"]');
    check("signup asks for a confirm-password field", passwords.length >= 2,
      `${passwords.length} password fields`);

    // Mismatched confirmation must be caught before any network call.
    await page.type('input[type="email"]', NEW_EMAIL);
    await passwords[0]?.type("supersecret");
    await passwords[1]?.type("different");
    await page.click('form button[type="submit"]');
    await sleep(400);
    check("mismatched passwords are rejected locally",
      /Passwords do not match/i.test(await text(page)));

    // Retype everything: the local check cleared nothing, but the fields still
    // hold the previous attempt's values.
    await clearLoginForm(page);
    await page.type('input[type="email"]', NEW_EMAIL);
    passwords = await page.$$('input[type="password"]');
    await passwords[0]?.type("supersecret");
    await passwords[1]?.type("supersecret");
    await page.click('form button[type="submit"]');
    check("signup creates an account and signs in", await waitForText(page, /START TRANSLATING/i));

    /* 8 — password reset --------------------------------------------------- */
    console.log("\nPassword reset:");
    // A fresh incognito context: sign-up left this page signed in and Firebase
    // persists that session, so the reset forms would not be reachable here.
    const resetCtx = await browser.createBrowserContext();
    const resetPage = await resetCtx.newPage();
    await resetPage.setViewport({ width: 414, height: 900, deviceScaleFactor: 1 });
    await installAuthBackend(resetPage, authLog, process.env.AUTH_STUB_DEBUG === "1");
    try {
      await resetPage.goto(`http://localhost:${PORT}/`, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await waitForText(resetPage, /Log in/i);
      check("switched to Forgot password", await switchMode(resetPage, "forgot"));
      await resetPage.waitForSelector('input[type="email"]');
      await resetPage.type('input[type="email"]', TEST_EMAIL);
      await resetPage.click('form button[type="submit"]');
      check("reset confirmation is shown",
        await waitForText(resetPage, /reset link is on its way/i));
      check("Firebase actually sent a reset email",
        authLog.resetEmails.some((e) => e.toLowerCase() === TEST_EMAIL),
        JSON.stringify(authLog.resetEmails));
    } finally {
      await resetCtx.close();
    }
  } finally {
    await browser.close();
    server.close();
  }

  if (failures) {
    console.error(`\nAUTH VERIFY FAILED: ${failures} check(s)`);
    process.exit(1);
  }
  console.log("\nAUTH VERIFY PASSED");
}

void main();
