/**
 * Load the normalized 399-class model from local files inside a real browser
 * and report the actual error (if any). No server required.
 *
 * Run:  npx tsx scripts/verify-model-local.ts <model.json> [weights.bin]
 */
import * as fs from "fs";
import * as path from "path";
import puppeteer, { type Browser } from "puppeteer-core";

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

function readMinified(pkgName: string): string {
  const dir = path.resolve("node_modules", pkgName, "dist");
  const candidates = fs.readdirSync(dir).filter((f) => f.endsWith(".js"));
  const min = candidates.find((f) => f.includes(".min.")) ?? candidates[0];
  if (!min) throw new Error(`No JS file in ${dir}`);
  return fs.readFileSync(path.join(dir, min), "utf8");
}

async function main() {
  const modelJsonPath = process.argv[2];
  if (!modelJsonPath) {
    console.error("Usage: node scripts/verify-model-local.ts <model.json> [weights.bin]");
    process.exit(2);
  }
  const binPath =
    process.argv[3] ?? path.join(path.dirname(modelJsonPath), "group1-shard1of1.bin");
  const modelJson = fs.readFileSync(modelJsonPath, "utf8");
  const binBuf = fs.readFileSync(binPath);

  let browser: Browser | null = null;
  try {
    browser = await puppeteer.launch({
      executablePath: findBrowser(),
      headless: true,
      args: ["--no-sandbox", "--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
    });
    const page = await browser.newPage();

    const logs: string[] = [];
    page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
    page.on("pageerror", (e: unknown) => logs.push(`[pageerror] ${(e as Error).message}`));

    // Inject tfjs + webgl backend from local node_modules (CDN is blocked)
    await page.evaluateOnNewDocument(readMinified("@tensorflow/tfjs"));
    await page.evaluateOnNewDocument(readMinified("@tensorflow/tfjs-backend-webgl"));

    // Serve model files via data URLs so no server is needed
    const jsonB64 = Buffer.from(modelJson).toString("base64");
    const binB64 = Buffer.from(binBuf).toString("base64");
    const jsonUrl = `data:application/json;base64,${jsonB64}`;
    const binUrl = `data:application/octet-stream;base64,${binB64}`;

    await page.evaluateOnNewDocument(
      (jsonUrl: string, binUrl: string) => {
        // @ts-ignore
        (window as any).__MODEL_JSON_URL = jsonUrl;
        // @ts-ignore
        (window as any).__MODEL_BIN_URL = binUrl;
        const origFetch = window.fetch.bind(window);
        window.fetch = (input: any, init?: any) => {
          const url = typeof input === "string" ? input : input?.url;
          if (url && url.endsWith("model.json")) {
            // @ts-ignore
            return fetch((window as any).__MODEL_JSON_URL);
          }
          if (url && (url.endsWith(".bin") || url.includes("group1-shard1of1"))) {
            // @ts-ignore
            return fetch((window as any).__MODEL_BIN_URL);
          }
          return origFetch(input, init);
        };
      },
      jsonUrl,
      binUrl,
    );

    await page.goto("about:blank", { waitUntil: "load" });
    await new Promise((r) => setTimeout(r, 500));

    const result = await page.evaluate(async () => {
      try {
        // @ts-ignore
        const tf = (window as any).tf;
        if (!tf) return "tf not on window";
        // CPU backend — headless WebGL is unavailable in this sandbox
        await tf.setBackend("cpu");
        await tf.ready();

        const m = await tf.loadLayersModel("/models/fsl_model/model.json");
        const t = tf.zeros([1, 16, 345]);
        const p = m.predict(t) as any;
        const shape = p.shape;
        p.dispose(); t.dispose(); m.dispose();
        return "LOADED OK shape=" + JSON.stringify(shape);
      } catch (e: unknown) {
        const err = e as Error;
        return "FAILED: " + JSON.stringify({
        message: err?.message || String(err),
        name: err?.name,
        stack: err?.stack?.split("\n").slice(0, 4).join(" | "),
      });
      }
    });

    console.log("MODEL RESULT:", result);
    console.log("CONSOLE LOGS:");
    console.log(logs.join("\n") || "(none)");
  } finally {
    if (browser) await browser.close();
  }
}

main().catch((e) => {
  console.error("DRIVER FAILED:", e instanceof Error ? e.message : String(e));
  process.exit(1);
});