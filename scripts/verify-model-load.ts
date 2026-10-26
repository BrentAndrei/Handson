/**
 * Loads the deployed FSL model directly against the LIVE Firebase site in a
 * real browser and reports the actual error (if any).
 *
 * Run:  npx tsx scripts/verify-model-load.ts
 */
import * as fs from "fs";
import * as path from "path";
import puppeteer, { type Browser } from "puppeteer-core";

const FIREBASEL_URL = "https://handson-f042a.web.app";
const MODEL_PATH = "/models/fsl_model/model.json";

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
    page.on("responsefailed", (r: unknown) => {
      const url = (r as any)?.url();
      const err = (r as any)?.failure();
      logs.push(`[resp-failed] ${url} ${err?.errorText}`);
    });

    // Inject tfjs + webgl backend from local node_modules (CDN is blocked)
    await page.evaluateOnNewDocument(readMinified("@tensorflow/tfjs"));
    await page.evaluateOnNewDocument(readMinified("@tensorflow/tfjs-backend-webgl"));

    await page.goto(FIREBASEL_URL, { waitUntil: "networkidle0", timeout: 60000 });
    await new Promise((r) => setTimeout(r, 3000));

    const result = await page.evaluate(
      async (modelPath: string) => {
        try {
          // @ts-ignore
          const tf = (window as any).tf;
          if (!tf) return "tf not on window";
          // CPU backend — headless WebGL is unavailable in this sandbox
          await tf.setBackend("cpu");
          await tf.ready();
          const m = await tf.loadLayersModel(modelPath);
          const t = tf.zeros([1, 16, 345]);
          const p = m.predict(t) as any;
          const shape = p.shape;
          p.dispose(); t.dispose(); m.dispose();
          return "LOADED OK shape=" + JSON.stringify(shape);
        } catch (e: any) {
          return "FAILED: " + JSON.stringify({
            message: e?.message || String(e),
            name: e?.name,
            stack: e?.stack?.split("\n").slice(0, 4).join(" | "),
          });
        }
      },
      MODEL_PATH,
    );

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