/**
 * PHASE 1 UI smoke check.
 *
 * Boots the built app in headless Edge and asserts the neo-brutalist home
 * screen actually renders: both feature cards, both CTAs, the bottom nav, and
 * four-item navigation. Also captures a screenshot for visual review.
 *
 * This is a UI-layer gate only. It deliberately asserts NOTHING about the
 * retargeting pipeline — verify:retarget / verify:golden own that.
 */
import { launch } from "puppeteer-core";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(process.cwd());
const DIST = join(ROOT, "dist");
const PORT = 4317;

const EDGE_CANDIDATES = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
];

const MIME: Record<string, string> = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".glb": "model/gltf-binary", ".wasm": "application/wasm", ".jsonl": "application/json",
};

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` \u2014 ${detail}` : ""}`);
}

async function main() {
  if (!existsSync(join(DIST, "index.html"))) {
    console.error("dist/index.html missing. Run `npm run build` first.");
    process.exit(1);
  }

  const server = createServer(async (req, res) => {
    const url = (req.url ?? "/").split("?")[0];
    let p = join(DIST, url === "/" ? "index.html" : decodeURIComponent(url));
    if (!existsSync(p)) p = join(DIST, "index.html"); // SPA fallback
    try {
      const body = await readFile(p);
      res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  await new Promise<void>((r) => server.listen(PORT, r));

  const exe = EDGE_CANDIDATES.find((p) => existsSync(p));
  if (!exe) {
    console.error("No Chromium-based browser found for the UI smoke check.");
    server.close();
    process.exit(1);
  }

  const browser = await launch({
    executablePath: exe,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu", "--use-gl=swiftshader"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 414, height: 900, deviceScaleFactor: 2 });

    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });

    await page.goto(`http://localhost:${PORT}/`, { waitUntil: "networkidle0", timeout: 60_000 });
    await new Promise((r) => setTimeout(r, 1200));

    const text = await page.evaluate(() => document.body.innerText);
    console.log("Home screen:");
    check("wordmark", /HandSon/i.test(text));
    check("greeting", /Hello/i.test(text));
    check("tagline", /Filipino Sign Language/i.test(text));
    check("card 1 heading", /REAL-TIME[\s\S]*TRANSLATOR/i.test(text));
    check("card 1 CTA", /START TRANSLATING/i.test(text));
    check("card 2 heading", /3D SIGN[\s\S]*AVATAR/i.test(text));
    check("card 2 CTA", /OPEN 3D AVATAR/i.test(text));
    check("no invented features", !/practice mode|learn mode|favourites|favorites/i.test(text));

    /* ------------------------------------------------------------------
       VISUAL IDENTITY GATES
       These assert the DESIGN LANGUAGE is present, not merely that a card
       exists. Each one encodes a rule the design brief depends on, so a
       future edit that quietly removes the layer fails the build instead of
       shipping something that still "looks like a dashboard".

       NOTE: the body is a STRING on purpose. tsx compiles this file with
       esbuild, which rewrites named inner functions to a `__name` helper that
       does not exist in the browser realm — passing a real function throws
       "ReferenceError: __name is not defined". A string literal is never
       transformed, so it runs verbatim in the page. Same as CONTRAST_SCAN.
       ------------------------------------------------------------------ */
    const IDENTITY_SCAN = `(() => {
      const scope = document.querySelector("[data-view='home']");
      if (!scope) return null;
      const cs = (el) => (el ? getComputedStyle(el) : null);

      // 1. Depth must be HARD: no box-shadow may carry a blur radius. The
      //    assertion is on the RATIO, not a raw count, because a design with
      //    3 hard shadows and 1 soft one is worse than one with 12 hard ones.
      const shadows = Array.from(scope.querySelectorAll("*"))
        .map((el) => getComputedStyle(el).boxShadow)
        .filter((s) => s && s !== "none");
      const hardShadows = shadows.filter((s) => !/blur/.test(s));
      const softShadows = shadows.filter((s) => /blur/.test(s));

      // 2. Sticker layer: rotated printed labels must exist.
      const stickers = scope.querySelectorAll(".hs-sticker");
      const rotated = Array.from(stickers).filter(
        (el) => cs(el).transform !== "none" && cs(el).transform !== "matrix(1, 0, 0, 1, 0, 0)"
      );

      // 3. Oversized display type must actually be oversized.
      const mega = Array.from(scope.querySelectorAll(".hs-mega, .hs-numeral"));
      const megaSizes = mega.map((el) => parseFloat(getComputedStyle(el).fontSize));
      const biggest = megaSizes.length ? Math.max.apply(null, megaSizes) : 0;

      // 4. Layering: elements that carry a transform (rotation / bleed).
      const bleeders = Array.from(scope.querySelectorAll("*")).filter((el) => {
        const t = getComputedStyle(el).transform;
        return t !== "none" && /matrix/.test(t);
      });

      // 5. Saturated colour blocking: large flat areas, not tiny accents.
      const slabs = Array.from(scope.querySelectorAll("div, article")).filter((el) => {
        const r = el.getBoundingClientRect();
        return r.width * r.height > 6000;
      });
      const coloured = slabs.filter((el) => {
        const bg = getComputedStyle(el).backgroundColor;
        const m = (bg.match(/\\d+/g) || []).slice(0, 3).map(Number);
        if (m.length < 3) return false;
        const mx = Math.max(m[0], m[1], m[2]);
        const mn = Math.min(m[0], m[1], m[2]);
        return mx > 120 && mx - mn > 55;
      });

      // 6. Both mascots present and large enough to read as illustrations.
      //    Selected by [data-mascot], NOT by a generic svg[viewBox]: the cards
      //    also contain sticker icons, and matching those made the asymmetry
      //    check below compare two icons instead of two mascots.
      const mascots = scope.querySelectorAll("[data-mascot]");
      const bigMascots = Array.from(mascots)
        .map((el) => el.getBoundingClientRect())
        .filter((r) => Math.round(r.width) > 90 && Math.round(r.height) > 60)
        .map((r) => Math.round(r.width) + "x" + Math.round(r.height));

      return {
        totalShadows: shadows.length,
        hardShadows: hardShadows.length,
        softShadows: softShadows.length,
        stickers: stickers.length,
        rotated: rotated.length,
        biggestMega: Math.round(biggest),
        bleeders: bleeders.length,
        colouredSlabs: coloured.length,
        mascotCount: mascots.length,
        bigMascots: bigMascots,
      };
    })()`;
    const identity = (await page.evaluate(IDENTITY_SCAN)) as {
      totalShadows: number; hardShadows: number; softShadows: number;
      stickers: number; rotated: number; biggestMega: number; bleeders: number;
      colouredSlabs: number; mascotCount: number; bigMascots: string[];
    } | null;

    console.log("Visual identity:");
    check("identity scan ran", identity !== null);
    check("all shadows are hard (zero blur)", (identity?.softShadows ?? 1) === 0, `${identity?.softShadows} soft of ${identity?.totalShadows}`);
    check("hard shadow depth is used", (identity?.hardShadows ?? 0) >= 4, String(identity?.hardShadows));
    check("sticker layer present", (identity?.stickers ?? 0) >= 4, String(identity?.stickers));
    check("stickers are rotated", (identity?.rotated ?? 0) >= 3, String(identity?.rotated));
    check("oversized display type", (identity?.biggestMega ?? 0) >= 56, (identity?.biggestMega ?? 0) + "px");
    check("layered / bleeding elements", (identity?.bleeders ?? 0) >= 3, String(identity?.bleeders));
    check("saturated colour blocking", (identity?.colouredSlabs ?? 0) >= 2, String(identity?.colouredSlabs));
    check("two mascots present", (identity?.mascotCount ?? 0) === 2, String(identity?.mascotCount));
    check("mascots are large illustrations", (identity?.bigMascots.length ?? 0) >= 2, JSON.stringify(identity?.bigMascots));

    // 7. Asymmetry: the two feature cards must NOT be mirror images. If the
    //    mascot sits on the same side of both, the design has collapsed into
    //    two identical panels — which is exactly what Phase 1 shipped.
    const ASYMMETRY_SCAN = `(() => {
      const cards = Array.from(document.querySelectorAll("[data-view='home'] article"));
      if (cards.length < 2) return null;
      const side = (card) => {
        const mascot = card.querySelector("[data-mascot]");
        if (!mascot) return "none";
        const cardBox = card.getBoundingClientRect();
        const m = mascot.getBoundingClientRect();
        return (m.left + m.width / 2) < (cardBox.left + cardBox.width / 2) ? "left" : "right";
      };
      return { a: side(cards[0]), b: side(cards[1]) };
    })()`;
    const asymmetry = (await page.evaluate(ASYMMETRY_SCAN)) as { a: string; b: string } | null;
    check(
      "feature cards break symmetry",
      asymmetry !== null && asymmetry.a !== asymmetry.b,
      asymmetry ? `card1=${asymmetry.a} card2=${asymmetry.b}` : "no cards"
    );

    // 8. No horizontal overflow — decoration must never cause sideways panning,
    //    which is the classic failure mode of deliberately offset elements.
    const OVERFLOW_SCAN = `(() => ({
      scroll: document.documentElement.scrollWidth,
      client: document.documentElement.clientWidth
    }))()`;
    const overflow = (await page.evaluate(OVERFLOW_SCAN)) as { scroll: number; client: number };
    check(
      "no horizontal overflow",
      overflow.scroll <= overflow.client + 1,
      `${overflow.scroll}px vs ${overflow.client}px`
    );

    /* ------------------------------------------------------------------
       MASCOT LIVENESS
       A mascot whose `transform` is invalid CSS is dropped by the browser
       WITHOUT any error: the computed value becomes "none" and the art
       silently collapses in place. That is exactly what happened here - both
       hands rendered stacked at the origin because `calc(-3deg + <length>)` is
       invalid. It passed every other check, because the element still existed,
       was still large, and still had the right colours.

       So we assert the transforms actually resolve, and that the two hands
       occupy DIFFERENT boxes (a mascot pair stacked on itself is a defect).
       ------------------------------------------------------------------ */
    const MASCOT_SCAN = `(() => {
      const report = {};
      for (const name of ["eye", "hands"]) {
        const host = document.querySelector('[data-mascot="' + name + '"]');
        if (!host) { report[name] = { missing: true }; continue; }
        const svgs = Array.from(host.querySelectorAll("svg"));
        let resolved = 0;
        for (const svg of svgs) {
          for (const g of Array.from(svg.querySelectorAll("g"))) {
            const t = getComputedStyle(g).transform;
            if (t && t !== "none") resolved++;
          }
        }
        const top = svgs[0] ? svgs[0].querySelector(":scope > g") : null;
        const parts = top
          ? Array.from(top.children).map((g) => {
              const b = g.getBoundingClientRect();
              return { x: Math.round(b.left), y: Math.round(b.top) };
            })
          : [];
        report[name] = { resolved: resolved, parts: parts };
      }
      return report;
    })()`;
    const mascots = (await page.evaluate(MASCOT_SCAN)) as Record<
      string,
      { missing?: boolean; resolved?: number; parts?: { x: number; y: number }[] }
    >;

    console.log("Mascots:");
    check("eye mascot present", !mascots.eye?.missing);
    check("hands mascot present", !mascots.hands?.missing);
    check("eye transform resolves", (mascots.eye?.resolved ?? 0) >= 1, String(mascots.eye?.resolved));
    check("hands transforms resolve", (mascots.hands?.resolved ?? 0) >= 2, String(mascots.hands?.resolved));
    const handParts = mascots.hands?.parts ?? [];
    check(
      "the two hands are not stacked",
      handParts.length === 2 &&
        (Math.abs(handParts[0].x - handParts[1].x) > 10 || Math.abs(handParts[0].y - handParts[1].y) > 10),
      JSON.stringify(handParts)
    );

    const stacked = await page.evaluate(() => {
      const cards = Array.from(document.querySelectorAll("article"));
      if (cards.length < 2) return false;
      const a = cards[0].getBoundingClientRect();
      const b = cards[1].getBoundingClientRect();
      return b.top >= a.bottom - 2;
    });
    check("feature cards stacked", stacked);

    const nav = await page.evaluate(() => {
      const el = document.querySelector('nav[aria-label="Primary"]');
      if (!el) return null;
      const buttons = Array.from(el.querySelectorAll("button"));
      return {
        count: buttons.length,
        labels: buttons.map((b) => (b.textContent ?? "").trim()),
        activeCount: buttons.filter((b) => b.getAttribute("aria-current") === "page").length,
        activeIsHome: buttons[0]?.getAttribute("aria-current") === "page",
      };
    });
    check("bottom nav present", nav !== null);
    check("nav has 4 items", nav?.count === 4, String(nav?.count));
    check("exactly one active item", nav?.activeCount === 1, String(nav?.activeCount));
    check("Home is active on load", nav?.activeIsHome === true);
    check(
      "nav labels",
      JSON.stringify(nav?.labels) === JSON.stringify(["Home", "Translate", "Avatar", "Account"]),
      JSON.stringify(nav?.labels)
    );

    const a11y = await page.evaluate(() => {
      const h1s = document.querySelectorAll("h1").length;
      const btns = Array.from(document.querySelectorAll("button"));
      const unlabelled = btns.filter((b) => {
        const hasText = (b.textContent ?? "").trim().length > 0;
        const labelled = b.getAttribute("aria-label") || b.getAttribute("aria-labelledby");
        return !hasText && !labelled;
      }).length;
      const heights = btns.map((b) => b.getBoundingClientRect().height).filter((h) => h > 0);
      return { h1s, unlabelled, minTouch: Math.round(Math.min(...heights)) };
    });
    check("single h1", a11y.h1s === 1, String(a11y.h1s));
    check("no unlabelled buttons", a11y.unlabelled === 0, String(a11y.unlabelled));
    check("touch targets >= 38px", a11y.minTouch >= 38, a11y.minTouch + "px");

    /* Navigation actually works: tapping Translate swaps the active item.

       IMPORTANT ORDERING: the home-view assertions and the screenshot below
       must run BEFORE this click, because every one of them is scoped to
       [data-view=home]. Screenshotting after navigation produced a capture of
       the Translate view labelled "ui-home.png", which is how a broken home
       screen can pass review. The click is therefore moved to the very end. */
    const navClick = async () => {
      await page.evaluate(() => {
        const navBtns = document.querySelectorAll('nav[aria-label="Primary"] li button');
        (navBtns[1] as HTMLElement)?.click();
      });
      await new Promise((r) => setTimeout(r, 600));
      const afterNav = await page.evaluate(
        () => document.querySelector('nav[aria-label="Primary"] li button[aria-current="page"]')?.textContent?.trim() ?? ""
      );
      check("nav switches to Translate", /Translate/i.test(afterNav), afterNav);
    };

    /* Contrast + surface checks. A DOM-only gate can pass while the page is
       still rendering the OLD dark theme, so these read computed styles. */
    const visual = await page.evaluate(() => {
      const body = getComputedStyle(document.body);
      const heading = document.querySelector("article h2");
      const hs = heading ? getComputedStyle(heading) : null;
      const card = document.querySelector("article");
      const cs = card ? getComputedStyle(card) : null;
      const root = getComputedStyle(document.documentElement);
      return {
        bodyBg: body.backgroundColor,
        bodyColor: body.color,
        headingColor: hs?.color ?? "",
        cardBg: cs?.backgroundColor ?? "",
        cardRadius: cs?.borderRadius ?? "",
        radiusToken: root.getPropertyValue("--radius").trim(),
        navBg: getComputedStyle(document.querySelector('nav[aria-label="Primary"]')!).backgroundColor,
      };
    });
    check("body is paper, not navy", !/rgb\(2, 6, 23\)|rgb\(12, 18, 34\)/.test(visual.bodyBg), visual.bodyBg);
    check("body text is dark ink", !/rgb\(241, 245, 249\)/.test(visual.bodyColor), visual.bodyColor);
    check("card heading is dark on light", !/rgb\(241, 245, 249\)/.test(visual.headingColor), visual.headingColor);
    check("radius token is small", visual.radiusToken === "4px", visual.radiusToken);
    check("card radius follows token", visual.cardRadius === "4px", visual.cardRadius);

    // No blur/backdrop-filter on the HOME screen. Scoped by CONTAINMENT in the
    // live [data-view="home"] subtree rather than "everything in <main>": the
    // translate/avatar views stay mounted inside AnimatePresence, so a naive
    // sweep would flag their (still-legacy, Phase 2) glass panels.
    const noBlur = await page.evaluate(() => {
      const scope = document.querySelector("[data-view='home']");
      if (!scope) return ["no [data-view=home] scope found"];
      const bad: string[] = [];
      for (const el of Array.from(scope.querySelectorAll("*"))) {
        const s = getComputedStyle(el);
        if (s.backdropFilter && s.backdropFilter !== "none") {
          bad.push("backdrop:" + el.tagName + "." + String(el.className).slice(0, 24));
        }
        if (s.filter && s.filter !== "none") bad.push("filter:" + el.tagName);
        if (s.boxShadow !== "none" && s.boxShadow.includes("blur")) {
          bad.push("blur-shadow:" + el.tagName);
        }
      }
      return bad;
    });
    check("no blur/backdrop-filter on home", noBlur.length === 0, noBlur.slice(0, 2).join(" | "));

    // Contrast sweep: no visible text may sit near-invisible on its own
    // background. Catches hardcoded `text-white` left over from the dark theme
    // now that the page is light.
    //
    // The body is passed to evaluate() as a STRING on purpose. tsx compiles
    // this file with esbuild, which rewrites named inner functions to a
    // `__name` helper that does not exist in the browser realm, throwing
    // "ReferenceError: __name is not defined". A string literal is never
    // transformed, so the code runs verbatim in the page.
    const CONTRAST_SCAN = `(() => {
      const bgOf = (el) => {
        let n = el;
        while (n && n !== document.documentElement) {
          const bg = getComputedStyle(n).backgroundColor;
          const a = (bg.match(/[\\d.]+/g) || [])[3];
          if (bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent' && (a === undefined || Number(a) > 0.85)) return bg;
          n = n.parentElement;
        }
        return getComputedStyle(document.body).backgroundColor;
      };
      const parts = (c) => (c.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
      const chan = (v) => (v / 255 <= 0.03928 ? v / 255 / 12.92 : Math.pow((v / 255 + 0.055) / 1.055, 2.4));
      const bad = [];
      const sel = '[data-view=home] span, [data-view=home] p, [data-view=home] h1, [data-view=home] h2, [data-view=home] h3, [data-view=home] button, [data-view=home] a, [data-view=home] li';
      for (const el of Array.from(document.querySelectorAll(sel))) {
        if (el.closest('nav[aria-label="Primary"]') || el.children.length > 0) continue;
        const text = (el.textContent || '').trim();
        if (!text) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width < 2 || rect.height < 2) continue;
        const cs = getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.opacity === '0') continue;
        const fg = parts(cs.color), bg = parts(bgOf(el));
        const l1 = 0.2126 * chan(fg[0]) + 0.7152 * chan(fg[1]) + 0.0722 * chan(fg[2]);
        const l2 = 0.2126 * chan(bg[0]) + 0.7152 * chan(bg[1]) + 0.0722 * chan(bg[2]);
        const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
        if (ratio < 4.5) bad.push(el.tagName + ' "' + text.slice(0, 22) + '" ' + ratio.toFixed(1) + ':1');
      }
      return bad;
    })()`;
    const lowContrast = (await page.evaluate(CONTRAST_SCAN)) as string[];
    check("text meets 4.5:1 contrast", lowContrast.length === 0, lowContrast.slice(0, 3).join(" | "));
    // The nav must be pinned to the viewport bottom, and the document must
    // reserve space for it so the LAST card is not permanently occluded.
    // (Content scrolling *behind* a fixed bar is normal and correct; what must
    // never happen is the final element resting underneath it at max scroll.)
    const navBox = await page.evaluate(() => {
      const el = document.querySelector('nav[aria-label="Primary"]')!;
      const r = el.getBoundingClientRect();
      return { bottomGap: Math.round(window.innerHeight - r.bottom), position: getComputedStyle(el).position };
    });
    check("nav pinned to viewport bottom", Math.abs(navBox.bottomGap) <= 2, `gap ${navBox.bottomGap}px, ${navBox.position}`);

    // Scroll to the very bottom and confirm the last content clears the nav.
    const atRest = await page.evaluate(async () => {
      window.scrollTo(0, document.documentElement.scrollHeight);
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const nav = document.querySelector('nav[aria-label="Primary"]')!.getBoundingClientRect();
      let maxBottom = 0;
      // Exclude the nav's own subtree — its children are legitimately flush
      // with the bar, and counting them guarantees a false failure.
      for (const el of Array.from(document.querySelectorAll("main *"))) {
        if (el.closest('nav[aria-label="Primary"]')) continue;
        const r = el.getBoundingClientRect();
        if (r.height > 0 && r.bottom <= window.innerHeight + 1) maxBottom = Math.max(maxBottom, r.bottom);
      }
      return Math.round(maxBottom - nav.top);
    });
    check("last content clears nav at full scroll", atRest <= 0, `overlap ${atRest}px`);
    await page.evaluate(() => window.scrollTo(0, 0));
    await new Promise((r) => setTimeout(r, 200));

    /* Full-page capture.
       The nav is `position: fixed`, and in a fullPage screenshot the browser
       paints it at the ORIGINAL viewport offset rather than at the bottom of
       the document — so it lands in the middle of the image and appears to cut
       the cards in half. Toggling it to `position: static` for the capture only
       moves it to the true end of the flow, which is what a reader sees. The
       change is reverted immediately afterwards, so no assertion and no runtime
       state is affected. */
    const NEUTRALISE_NAV = `(() => {
      const nav = document.querySelector('nav[aria-label="Primary"]');
      if (nav) { nav.dataset.prevPos = nav.style.position || ""; nav.style.position = "static"; }
      return true;
    })()`;
    const RESTORE_NAV = `(() => {
      const nav = document.querySelector('nav[aria-label="Primary"]');
      if (nav) { nav.style.position = nav.dataset.prevPos || ""; }
      return true;
    })()`;

    await mkdir(join(ROOT, "tmp"), { recursive: true });
    await page.evaluate(NEUTRALISE_NAV);
    await new Promise((r) => setTimeout(r, 250));
    await page.screenshot({ path: join(ROOT, "tmp", "ui-home.png"), fullPage: true });
    await page.evaluate(RESTORE_NAV);
    console.log("\n  screenshot -> tmp/ui-home.png");

    /* Desktop capture. The design deliberately allows MORE asymmetry and
       oversized type at wide widths, so a mobile-only capture cannot verify
       it. The mascot states are captured too: with no pointer, both rest
       centred, which is exactly what should be reviewed. */
    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await new Promise((r) => setTimeout(r, 700));
    await page.evaluate(NEUTRALISE_NAV);
    await new Promise((r) => setTimeout(r, 250));
    await page.screenshot({ path: join(ROOT, "tmp", "ui-desktop.png"), fullPage: true });
    await page.evaluate(RESTORE_NAV);
    console.log("  screenshot -> tmp/ui-desktop.png");

    // Back to the phone viewport so the nav-clearance check below is measured
    // against the same geometry the earlier assertions used.
    await page.setViewport({ width: 414, height: 900, deviceScaleFactor: 2 });
    await new Promise((r) => setTimeout(r, 500));

    const fatal = errors.filter((e) => !/favicon|manifest|sw\.js|404|Failed to load resource/i.test(e));
    check("no console/page errors", fatal.length === 0, fatal.slice(0, 2).join(" | "));

    // Navigation check runs LAST, once every home-scoped assertion is done.
    await navClick();

    /* REGRESSION: only one view may be mounted at a time.
     *
     * The home view used to render as a plain <div> with no `key`, so
     * AnimatePresence (mode="wait") never unmounted it. Navigating to
     * Translate left the home hero on screen and pushed the translate view
     * ~1650px below the fold, so clicking TRANSLATE appeared to do nothing.
     * Assert that the home view is GONE once another view is active, and that
     * the new view actually sits inside the viewport. */
    await new Promise((r) => setTimeout(r, 900));
    const viewState = (await page.evaluate(`(() => {
      const home = document.querySelector('[data-view="home"]');
      const back = Array.from(document.querySelectorAll("button"))
        .find((e) => /Back/i.test(e.textContent || ""));
      return {
        homePresent: Boolean(home),
        backTop: back ? Math.round(back.getBoundingClientRect().top) : null,
        vh: window.innerHeight,
      };
    })()`)) as { homePresent: boolean; backTop: number | null; vh: number };
    check("previous view unmounts after navigation",
      !viewState.homePresent,
      `home still mounted: ${viewState.homePresent}`);
    check("navigated view is within the viewport",
      viewState.backTop !== null && viewState.backTop >= 0 && viewState.backTop < viewState.vh,
      `top=${viewState.backTop}px, viewport=${viewState.vh}px`);
    // UI_CHECK_MARKER
  } finally {
    await browser.close();
    server.close();
  }

  if (failures) {
    console.error(`\nUI VERIFY FAILED: ${failures} check(s)`);
    process.exit(1);
  }
  console.log("\nUI VERIFY PASSED");
}

void main();
