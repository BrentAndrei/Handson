import { cn } from "../../lib/cn";

type Tone = "ink" | "lime" | "sun" | "teal" | "coral" | "pink" | "paper";

const TONE: Record<Tone, string> = {
  ink: "bg-[var(--ink)] text-[var(--paper)]",
  lime: "bg-[var(--lime)] text-[var(--ink)]",
  sun: "bg-[var(--sun)] text-[var(--ink)]",
  teal: "bg-[var(--teal)] text-[var(--ink)]",
  coral: "bg-[var(--coral)] text-[var(--ink)]",
  pink: "bg-[var(--pink)] text-[var(--ink)]",
  paper: "bg-[var(--paper-sunken)] text-[var(--ink)]",
};

export interface BrutalBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

/**
 * A state or category chip: flat fill, hard 2px border, no radius to speak of.
 *
 * Used for STATE and CATEGORISATION only. It is never the sole carrier of
 * meaning — whatever it labels also states the same thing in words, because a
 * colour-only signal is invisible to a colourblind user.
 */
export function BrutalBadge({ tone = "ink", className, children, ...rest }: BrutalBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-[var(--radius-sm)]",
        "border-[var(--border-w-thin)] border-[var(--ink)]",
        "px-2 py-0.5",
        "font-[family-name:var(--font-display)] text-[10px] uppercase leading-none tracking-[0.12em]",
        TONE[tone],
        className
      )}
      {...rest}
    >
      {children}
    </span>
  );
}