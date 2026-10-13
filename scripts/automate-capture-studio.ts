/**
 * Automate Capture Studio Import tab via Puppeteer.
 * 
 * Usage:
 *   tsx scripts/automate-capture-studio.ts
 * 
 * Prerequisites:
 *   - Dev server running on localhost:5173 (npm run dev)
 *   - Videos in handson/training/youtube_raw/<LABEL>/
 *   - User already signed in (or use test auth)
 */

import { launch, type Page, type Browser } from "puppeteer-core";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { installAuthBackend, signInViaUi, authStubLog, clickByText } from "./firebase-test-backend";

const ROOT = process.cwd();
const YOUTUBE_RAW = join(ROOT, "training", "youtube_raw");
const DEV_URL = "https://localhost:5173";

const EDGE_CANDIDATES = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
];

interface VideoFile {
  label: string;
  path: string;
  name: string;
}

async function collectVideos(): Promise<VideoFile[]> {
  const videos: VideoFile[] = [];
  const labels = await import("node:fs/promises").then(fs => fs.readdir(YOUTUBE_RAW));
  
  for (const label of labels) {
    const labelDir = join(YOUTUBE_RAW, label);
    const stat = await import("node:fs/promises").then(fs => fs.stat(labelDir));
    if (!stat.isDirectory()) continue;
    
    const files = await import("node:fs/promises").then(fs => fs.readdir(labelDir));
    for (const file of files) {
      if (file.toLowerCase().endsWith(".mp4")) {
        videos.push({
          label,
          path: join(labelDir, file),
          name: file
        });
      }
    }
  }
  return videos;
}

async function waitForSelectorSafe(page: Page, selector: string, timeout = 10000): Promise<boolean> {
  try {
    await page.waitForSelector(selector, { timeout });
    return true;
  } catch {
    return false;
  }
}

async function main() {
  console.log("Collecting videos from youtube_raw...");
  const videos = await collectVideos();
  console.log(`Found ${videos.length} videos across ${new Set(videos.map(v => v.label)).size} labels`);
  
  for (const v of videos) {
    console.log(`  ${v.label}: ${v.name}`);
  }

  const exe = EDGE_CANDIDATES.find(p => existsSync(p));
  if (!exe) {
    console.error("No Edge/Chrome executable found");
    process.exit(1);
  }

  console.log(`\nLaunching browser: ${exe}`);
const browser = await launch({
    executablePath: exe,
    headless: false,
    ignoreHTTPSErrors: true,
    args: [
      "--no-sandbox", 
      "--disable-setuid-sandbox", 
      "--use-fake-ui-for-media-stream",
      "--ignore-certificate-errors",
      "--ignore-certificate-errors-spki-list",
      "--ignore-ssl-errors"
    ],
    defaultViewport: { width: 1400, height: 900 },
  });

  const page = await browser.newPage();
  
  // Enable console logging from browser
  page.on("console", msg => {
    if (msg.type() === "error") {
      console.log(`  [BROWSER ERROR] ${msg.text()}`);
    }
  });

  console.log(`Navigating to ${DEV_URL}...`);
  
  // Install Firebase auth backend BEFORE navigation
  const authLog = authStubLog();
  await installAuthBackend(page, authLog, true);
  
  await page.goto(DEV_URL, { waitUntil: "networkidle0", timeout: 60000 });

  // Wait for app to load
  await waitForSelectorSafe(page, "[data-view], form", 30000);
  console.log("App loaded");

  // Check if we're on login screen and sign in automatically
  const isLogin = await page.evaluate(() => {
    return document.querySelector('input[type="email"]') !== null;
  });
  
  if (isLogin) {
    console.log("Login screen detected - signing in automatically...");
    await signInViaUi(page);
    
    // Wait for app to transition after sign-in
    await waitForSelectorSafe(page, "[data-view], nav, main", 30000);
    console.log("Main app loaded after sign-in");
    
    // Debug: check what view we're in
    const currentView = await page.evaluate(() => {
      const el = document.querySelector('[data-view]');
      return el ? el.getAttribute('data-view') : 'none';
    });
    console.log(`Current view: ${currentView}`);
    
    // Also dump nav buttons
    const navButtons = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button, a[role="tab"], nav button'));
      return buttons.map(b => ({ text: b.textContent?.trim(), 'data-view': b.getAttribute('data-view') }));
    });
    console.log("Nav elements:", navButtons);
  }

  // Navigate to Import tab (VideoCaptureScreen with Import mode)
  // The nav uses BrutalNav with AppView type
  console.log("Looking for Import navigation...");
  
  // Use clickByText which does a proper DOM click inside the page context
  const importClicked = await clickByText(page, /^import$/i);
  if (importClicked) {
    console.log("Clicked Import nav via clickByText");
  } else {
    console.log("Import nav not found via clickByText");
  }
  
  // Wait for view to change
  await new Promise(r => setTimeout(r, 2000));
  
  // Debug: check what view we're in after clicking Import
  const viewAfterImport = await page.evaluate(() => {
    const el = document.querySelector('[data-view]');
    return el ? el.getAttribute('data-view') : 'none';
  });
  console.log(`View after Import click: ${viewAfterImport}`);
  
  // Debug: dump body to see Import UI
  const importHtml = await page.evaluate(() => document.body.innerHTML.substring(0, 5000));
  console.log("Import UI HTML preview:", importHtml.substring(0, 2000));

  // Wait for Import UI to render
  await waitForSelectorSafe(page, 'input[type="file"]', 30000);
  console.log("Import UI ready");

  // Process each video
  for (let i = 0; i < videos.length; i++) {
    const video = videos[i];
    console.log(`\n[${i + 1}/${videos.length}] Processing ${video.label} - ${video.name}`);
    
    try {
      await processVideo(page, video);
    } catch (err) {
      console.error(`  ERROR processing ${video.name}:`, err);
    }
  }

  console.log("\nAll videos processed. Exporting dataset...");
  await exportDataset(page);
  
  await browser.close();
  console.log("Done!");
}

async function processVideo(page: Page, video: VideoFile) {
  // Find file input
  const fileInput = await page.$('input[type="file"]');
  if (!fileInput) {
    throw new Error("File input not found");
  }

  // Upload video
  await fileInput.uploadFile(video.path);
  console.log(`  Uploaded: ${video.name}`);

  // Wait for video to load in player
  await waitForSelectorSafe(page, 'video', 15000);
  console.log("  Video loaded in player");

  // TODO: Mark in/out points
  // This requires human interaction or heuristic
  // For now, pause for manual marking
  console.log("  >>> MANUAL: Mark in/out points in browser, then press Enter here...");
  await new Promise(r => process.stdin.once("data", r));

  // Click Extract button
  const extractBtn = await page.$('button:has-text("Extract"), button[data-action="extract"]');
  if (extractBtn) {
    await extractBtn.click();
    console.log("  Extraction started");
    
    // Wait for extraction to complete
    await waitForSelectorSafe(page, '.extraction-complete, .clip-result', 60000);
    console.log("  Extraction complete");
  }

  // Test clip against model
  const testBtn = await page.$('button:has-text("Test"), button[data-action="test"]');
  if (testBtn) {
    await testBtn.click();
    console.log("  Testing clip...");
    await waitForSelectorSafe(page, '.test-result', 30000);
    console.log("  Test complete");
  }

  // Add to dataset
  const addBtn = await page.$('button:has-text("Add to dataset"), button[data-action="add"]');
  if (addBtn) {
    await addBtn.click();
    console.log("  Added to dataset");
  }
}

async function exportDataset(page: Page) {
  const exportBtn = await page.evaluateHandle(() => {
    const xpath = '//button[contains(text(), "Export")] | //button[@data-action="export"]';
    const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
    return result.singleNodeValue;
  });
  const exportEl = await exportBtn.asElement();
  if (exportEl) {
    await exportEl.click();
    console.log("Export triggered");
    
    // Wait for download or confirmation
    await waitForSelectorSafe(page, '.export-complete, .download-ready', 30000);
    console.log("Export complete");
  } else {
    console.log("Export button not found");
  }
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});