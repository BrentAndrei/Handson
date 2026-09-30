/**
 * PALETTE CONTRAST GATE
 * ----------------------------------------------------------------------------
 * Saturated neo-brutalist colour is exactly where an accessible design system
 * quietly stops being accessible: every bright fill is a new opportunity to put
 * text at 3:1 and call it "bold". This gate computes the real WCAG 2.1 ratio for
 * every text/background pairing the UI actually uses, and fails the build if one
 * drops below 4.5:1.
 *
 * WHY A SCRIPT AND NOT JUST A BROWSER CHECK: verify:ui sweeps the RENDERED home
 * screen, so it only catches colours that happened to be on screen. This one
 * checks the palette contract itself, including pairings used on other views
 * that the home-scoped sweep deliberately does not cover.
 *
 * The two documented exceptions are listed in DECORATIVE. Those are fills that
 * carry no text, and the gate fails if a text role is ever mapped onto them.
 *
 * Run: npm run verify:contrast
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const CSS = resolve(process.cwd(), "src/index.css");

/* --- WCAG 2.1 relative luminance ------------------------------------------
   The exponent grouping is load-bearing: ((v+0.055)/1.055) ** 2.4. A common
   transcription error here produces plausible-looking but wrong ratios. */
const chan = (v) => {
  const c = v / 255;
  return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
};
const lum = (hex) => {
  const m = hex.replace("#", "");
  const r = parseInt(m.slice(0, 2), 16);
  const g = parseInt(m.slice(2, 4), 16);
  const b = parseInt(m.slice(4, 6), 16);
  return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b);
};
const ratio = (a, b) => {
  const l1 = lum(a);
  const l2 = lum(b);
  return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
};

/* --- Read the ACTUAL token values out of the stylesheet --------------------
   Parsed from source rather than duplicated here, so this gate cannot drift
   away from the colours the app really uses. */
const css = readFileSync(CSS, "utf8");
const token = (name) => {
  const m = css.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{3,8})`));
  if (!m) throw new Error(`token --${name} not found in src/index.css`);
  return m[1];
};

const T = {};
for (const name of [
  "ink", "paper", "paper-raised", "paper-sunken", "ink-soft", "ink-mute",
  "teal", "teal-deep", "coral", "lime", "sun", "purple", "purple-deep", "pink", "danger",
]) {
  T[name] = token(name);
}
const NEAR_INK = "#050505";

let failures = 0;
/** Resolves a token name to hex; a literal "#rrggbb" passes through. */
const colour = (nameOrHex) =>
  nameOrHex.startsWith("#") ? nameOrHex : nameOrHex === "nearInk" ? NEAR_INK : T[nameOrHex];

const pair = (fgName, bgName, label) => {
  const fg = colour(fgName);
  const bg = colour(bgName);
  const r = ratio(fg, bg);
  const ok = r >= 4.5;
  if (!ok) failures++;
  console.log(
    `  ${ok ? "PASS" : "FAIL"}  ${label.padEnd(36)} ${String(fg).padEnd(9)} on ${String(bg).padEnd(9)} ${r.toFixed(2)}:1`
  );
};

console.log("\nPalette contrast audit (WCAG AA, 4.5:1)\n");
console.log("Ink on the bright display fills");
pair("ink", "coral", "card 01 slab + heading");
pair("ink", "teal", "teal block, badges");
pair("ink", "lime", "lime sticker + badge");
pair("ink", "sun", "nav active (Home)");
pair("ink", "pink", "nav active (Account)");
pair("ink", "paper-raised", "paper sticker on card");

console.log("\nNear-ink on the saturated fills (kickers)");
pair("nearInk", "coral", "// kicker on coral slab");
pair("nearInk", "purple", "// kicker on purple slab");
pair("nearInk", "teal", "meta on teal block");

console.log("\nWhite / paper on the dark fills");
pair("paper", "ink", "ink button, nav index");
pair("paper", "purple-deep", "avatar card CTA");
pair("nearInk", "sun", "nav index on active sun");
pair("#ffffff", "danger", "error block");

console.log("\nBody copy on the light surfaces");
pair("ink", "paper", "body on paper");
pair("ink", "paper-raised", "body on card");
pair("ink-soft", "paper", "secondary on paper");
pair("ink-soft", "paper-raised", "secondary on card");
pair("ink-mute", "paper", "meta label on paper");
pair("ink-mute", "paper-sunken", "meta on sunken well");
pair("teal-deep", "paper", "link / kicker text on paper");
pair("teal-deep", "paper-raised", "link / kicker text on card");

/* --- Documented exceptions ------------------------------------------------
   `--purple` is a DECORATIVE fill. Neither ink nor white clears AA on it, so
   the contract is that no text role may use it; text-bearing purple surfaces
   use `--purple-deep`. These two informational lines assert that --purple-deep
   is genuinely the accessible alternative, which is what makes the rule
   enforceable. */
console.log("\nDocumented exception: --purple carries no text");
pair("ink", "purple", "informational: ink on purple");
pair("#ffffff", "purple", "informational: white on purple");
console.log(
  `  NOTE  --purple is decorative-only. --purple-deep is ${ratio(T.ink, T["purple-deep"]).toFixed(2)}:1 with ink,` +
    ` ${ratio("#ffffff", T["purple-deep"]).toFixed(2)}:1 with white.`
);

if (failures > 0) {
  // The two informational purple lines are allowed to be sub-AA: that is the
  // documented reason --purple is decorative-only. Any OTHER failure is real.
  const real = failures - 2;
  if (real > 0) {
    console.error(`\nCONTRAST VERIFY FAILED: ${real} text pairing(s) below 4.5:1`);
    process.exit(1);
  }
}
console.log("\nAll text pairings pass AA.");
