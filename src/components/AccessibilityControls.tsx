import { BrutalLabel } from "./ui/BrutalLabel";

interface Props {
  textSize: "sm" | "base" | "lg";
  onTextSize: (size: "sm" | "base" | "lg") => void;
  highContrast: boolean;
  onHighContrast: (on: boolean) => void;
}

/**
 * An unobtrusive accessibility settings cluster.
 *
 * HandsOn's primary users are deaf and hard-of-hearing signers, so readable
 * text and strong contrast matter more than a decorative toggle. These
 * controls live at the bottom-right of the main content area, out of the
 * way of the camera feed and prediction hero text, but reachable without
 * hunting. They scroll with the page to guarantee they never overlap
 * other content.
 *
 * Both toggles are real state, not just visual chrome: `textSize` rescales
 * the whole app via a CSS attribute selector, and `highContrast` swaps the
 * glass panels to opaque dark surfaces with white text.
 */
/**
 * The accessibility cluster, rendered as a PRINTED PANEL rather than a row of
 * floating chips.
 *
 * These controls matter more here than in most apps — HandSon's primary users
 * are deaf and hard-of-hearing signers, so text size and contrast are
 * functional requirements rather than nice-to-haves. Giving them the same
 * border/shadow language as the cards says "this is a real control", and stops
 * them reading as leftover default chrome in the corner of a designed page.
 *
 * State is never colour-only: the active size carries `aria-pressed`, an ink
 * fill, AND a printed label; the contrast toggle states its state in words.
 */
const SIZES: { id: "sm" | "base" | "lg"; label: string; glyph: string }[] = [
  { id: "sm", label: "Small text", glyph: "A" },
  { id: "base", label: "Default text size", glyph: "Aa" },
  { id: "lg", label: "Large text", glyph: "Aaa" },
];

const chip =
  "flex min-h-[38px] items-center justify-center gap-1.5 rounded-[var(--radius-sm)] " +
  "border-[var(--border-w-thin)] border-[var(--ink)] " +
  "font-[family-name:var(--font-display)] uppercase tracking-[0.06em] " +
  "shadow-[var(--shadow-pressed)] " +
  "transition-[transform,box-shadow,background-color] duration-[var(--press-ms)] ease-[var(--ease-snap)] " +
  "hover:-translate-y-[1px] hover:shadow-[var(--shadow-sm)] " +
  "active:translate-x-[1px] active:translate-y-[1px] active:shadow-[var(--shadow-pressed)] " +
  "motion-reduce:transition-none motion-reduce:hover:translate-y-0";

export function AccessibilityControls({ textSize, onTextSize, highContrast, onHighContrast }: Props) {
  return (
    <section
      aria-label="Accessibility settings"
      className="relative mt-6 border-[var(--border-w)] border-[var(--ink)] bg-[var(--paper-raised)] p-3 shadow-[var(--shadow)]"
    >
      {/* Printed heading, rotated slightly, so the panel reads as pasted on
          rather than as a system widget. */}
      <BrutalLabel slashed tone="mute" className="absolute -top-2.5 left-3 bg-[var(--paper-raised)] px-1.5">
        Display
      </BrutalLabel>

      <div className="mt-1 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        {/* Text size. The visible glyphs stay A / Aa / Aaa, but each button
            carries an explicit aria-label so the control is not announced as
            bare letters, and every target is >= 38px. */}
        <div className="flex items-center gap-1.5" role="group" aria-label="Text size">
          {SIZES.map(({ id, label, glyph }) => {
            const active = textSize === id;
            return (
              <button
                key={id}
                onClick={() => onTextSize(id)}
                className={`${chip} px-2.5 ${
                  active ? "bg-[var(--ink)] text-[var(--paper)]" : "bg-[var(--paper-sunken)] text-[var(--ink)] hover:bg-[var(--lime)]"
                } text-[13px]`}
                aria-pressed={active}
                aria-label={label}
              >
                <span className={active ? "" : "font-extrabold"}>{glyph}</span>
              </button>
            );
          })}
        </div>

        <button
          onClick={() => onHighContrast(!highContrast)}
          className={`${chip} px-3 py-1.5 text-[11px] ${
            highContrast
              ? "bg-[var(--sun)] text-[var(--ink)]"
              : "bg-[var(--paper-sunken)] text-[var(--ink)] hover:bg-[var(--lime)]"
          }`}
          aria-pressed={highContrast}
        >
          <span
            className="size-2.5 border-[var(--border-w-thin)] border-[var(--ink)] bg-current"
            aria-hidden="true"
          />
          {highContrast ? "Contrast: high" : "Contrast: normal"}
        </button>
      </div>
    </section>
  );
}
