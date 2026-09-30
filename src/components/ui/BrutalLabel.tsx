import { cn } from "../../lib/cn";

export interface BrutalLabelProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Renders the `//` editorial prefix used for section kickers. */
  slashed?: boolean;
  tone?: "ink" | "soft" | "mute" | "paper" | "ink-on-bright";
}

/**
 * Tiny uppercase metadata — the small half of the scale contrast.
 *
 * Always sits at 10-11px with wide tracking, directly against a huge heading.
 * A label that is not paired with something much larger is just small text, so
 * this is used only in editorial positions.
 */
export function BrutalLabel({
  slashed = false,
  tone = "ink",
  className,
  children,
  ...rest
}: BrutalLabelProps) {
  const TONE = {
    ink: "text-[var(--ink)]",
    soft: "text-[var(--ink-soft)]",
    mute: "text-[var(--ink-mute)]",
    paper: "text-[var(--paper)]",
    "ink-on-bright": "text-[var(--ink)]",
  }[tone];

  return (
    <span
      className={cn(
        "m-0 font-[family-name:var(--font-display)] text-[10px] uppercase",
        "leading-none tracking-[0.24em]",
        TONE,
        className
      )}
      {...rest}
    >
      {/* The `//` is its own element so the contrast sweep measures it. It is
          always FULL ink, never the label's muted tone: a decorative glyph is
          still text, and at 10px a "faint" grey fails WCAG AA. Visual subtlety
          comes from size and spacing here, never from low contrast.

          `ink-on-bright` pairs with the saturated slabs (coral, purple, teal).
          On purple, pure #111 measures 4.3:1 — just under AA — so that role
          drops to near-black, which clears 5.1:1 on every bright fill while
          still reading as the same colour family. */}
      {slashed ? (
        <span
          aria-hidden="true"
          className={
            tone === "ink-on-bright"
              ? "text-[#050505]"
              : tone === "paper"
              ? "text-[var(--paper)]"
              : "text-[var(--ink)]"
          }
        >
          //&nbsp;
        </span>
      ) : null}
      {children}
    </span>
  );
}

/** A short ink rule, optionally labelled — the editorial "kicker" line. */
export function BrutalRule({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn("block h-[3px] w-10 bg-[var(--ink)]", className)}
    />
  );
}

/** Registration crosses and corner ticks. Always decorative. */
export function PrintMarks({ className }: { className?: string }) {
  return (
    <span aria-hidden="true" className={cn("pointer-events-none absolute inset-0", className)}>
      <span className="hs-mark-cross left-2 top-2" />
      <span className="hs-mark-cross right-2 top-2" />
      <span className="hs-mark-cross bottom-2 left-2" />
      <span className="hs-mark-cross bottom-2 right-2" />
    </span>
  );
}
