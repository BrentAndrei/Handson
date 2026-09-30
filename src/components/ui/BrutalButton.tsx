import { forwardRef } from "react";
import { cn } from "../../lib/cn";
import { IconArrowRight } from "../icons";

type Tone = "teal" | "coral" | "lime" | "sun" | "purple" | "ink" | "paper";
type Size = "sm" | "md" | "lg" | "xl";

const TONE: Record<Tone, string> = {
  teal: "bg-[var(--teal)] text-[var(--ink)]",
  coral: "bg-[var(--coral)] text-[var(--ink)]",
  lime: "bg-[var(--lime)] text-[var(--ink)]",
  sun: "bg-[var(--sun)] text-[var(--ink)]",
  purple: "bg-[var(--purple-deep)] text-white",
  ink: "bg-[var(--ink)] text-[var(--paper)]",
  paper: "bg-[var(--paper-raised)] text-[var(--ink)]",
};

const SIZE: Record<Size, string> = {
  // min-h is the touch target: 44px minimum on md/lg per WCAG 2.5.5.
  sm: "min-h-[38px] px-3 py-1.5 text-xs",
  md: "min-h-[44px] px-4 py-2 text-sm",
  lg: "min-h-[52px] px-6 py-3 text-base",
  // The oversized primary CTA. Uppercase, wide, unmissable.
  xl: "min-h-[60px] px-7 py-4 text-base",
};

export interface BrutalButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: Tone;
  size?: Size;
  /** Renders a right-arrow glyph; use on primary navigation actions. */
  withArrow?: boolean;
  block?: boolean;
}

/**
 * The tactile primitive. This is the single most important interaction in the
 * whole system, because tactile feedback is what separates a printed object
 * from a flat rectangle with a border.
 *
 * The three states, exactly as specified:
 *   rest  — sits on the paper with a hard 5px offset shadow
 *   hover — rises: translate(-2px, -2px), shadow grows to 7px. The object
 *           lifts TOWARD the viewer, and the deeper shadow is what sells it.
 *   press — sinks: translate(+3px, +3px), shadow collapses to a 1px stub.
 *           The key has travelled into the page and cannot go further.
 *
 * The shadow offsets and the translate are the SAME number, so the ink edge of
 * the shadow never separates from the button edge.
 */
export const BrutalButton = forwardRef<HTMLButtonElement, BrutalButtonProps>(
  function BrutalButton(
    { tone = "teal", size = "md", withArrow, block, className, children, ...rest },
    ref
  ) {
    return (
      <button
        ref={ref}
        type="button"
        className={cn(
          "group/btn relative inline-flex select-none items-center justify-center gap-2",
          "border-[var(--border-w)] border-[var(--ink)]",
          "rounded-[var(--radius)]",
          "font-[family-name:var(--font-display)] uppercase leading-none tracking-[0.08em]",
          "shadow-[var(--shadow)]",
          "transition-[transform,box-shadow,background-color] duration-[var(--press-ms)] ease-[var(--ease-snap)]",
          "hover:-translate-x-[2px] hover:-translate-y-[2px] hover:shadow-[var(--shadow-lift)]",
          "active:translate-x-[3px] active:translate-y-[3px] active:shadow-[var(--shadow-pressed)]",
          "motion-reduce:transition-none motion-reduce:hover:translate-x-0 motion-reduce:hover:translate-y-0",
          "disabled:pointer-events-none disabled:opacity-45 disabled:shadow-[var(--shadow-sm)]",
          TONE[tone],
          SIZE[size],
          block && "w-full",
          className
        )}
        {...rest}
      >
        <span>{children}</span>
        {withArrow ? <IconArrowRight className="h-[1.35em] w-[1.35em] shrink-0" aria-hidden /> : null}
      </button>
    );
  }
);