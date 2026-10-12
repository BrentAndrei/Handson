/**
 * A moving ink-on-sun type strip — the marquee is the signature
 * device of the neo-brutalist brief (the `--sun` token is
 * documented as "caution, marquee" for exactly this).
 *
 * MECHANICS
 * ---------
 * The item list is rendered TWICE back to back and the inner
 * track animates `translateX(-50%)`. Because every item carries
 * its own horizontal margin (and the track has no flex gap),
 * the two halves are exactly equal width, so the -50% point is
 * a perfect seam and the loop never jumps.
 *
 * SAFETY
 * ------
 *  - The strip is `overflow-hidden`, so the wide track can never
 *    widen the document (the UI gate asserts no horizontal
 *    overflow).
 *  - `aria-hidden`: it is decorative repetition of labels that
 *    already appear in the nav.
 *  - Ink on `--sun` is an approved pairing (6.2:1+), so the
 *    contrast sweep clears it at any size.
 *  - The animation is CSS, so the global `prefers-reduced-motion`
 *    rule freezes it into a static strip.
 */
const ITEMS = [
  "Real-time FSL",
  "3D sign avatar",
  "Sign practice",
  "Video import",
  "Works offline",
] as const;

export function MarqueeStrip() {
  return (
    <div
      aria-hidden="true"
      className="relative overflow-hidden border-y-[var(--border-w)] border-[var(--ink)] bg-[var(--sun)] py-2.5"
    >
      <div className="hs-marquee flex w-max items-center">
        {[...ITEMS, ...ITEMS].map((label, i) => (
          <span
            key={i}
            className="mx-5 flex items-center gap-5 whitespace-nowrap font-[family-name:var(--font-display)] text-[11px] uppercase leading-none tracking-[0.22em] text-[var(--ink)]"
          >
            {label}
            {/* Ink diamond separator — geometry, not a glyph, so it
                renders identically with any font stack. */}
            <span className="inline-block h-[7px] w-[7px] rotate-45 border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--ink)]" />
          </span>
        ))}
      </div>
    </div>
  );
}
