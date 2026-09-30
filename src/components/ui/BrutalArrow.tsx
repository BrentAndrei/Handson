import { cn } from "../../lib/cn";

/**
 * A chunky directional arrow used as a graphic device.
 *
 * Drawn as a filled path rather than a stroked line so it can carry an ink
 * outline and a hard shadow, matching the rest of the system. Decorative by
 * default; the button label already carries the meaning.
 */
export function BrutalArrow({
  className,
  direction = "right",
  size = 28,
  title,
}: {
  className?: string;
  direction?: "right" | "down" | "up-right";
  size?: number;
  title?: string;
}) {
  const rotate = direction === "down" ? 90 : direction === "up-right" ? -45 : 0;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      aria-hidden={title ? undefined : true}
      role={title ? "img" : undefined}
      focusable="false"
      className={cn("shrink-0", className)}
      style={{ transform: `rotate(${rotate}deg)` }}
    >
      {title ? <title>{title}</title> : null}
      {/* Hard shadow first, then the arrow itself. */}
      <path
        d="M4 13h14V5l12 11-12 11v-8H4z"
        fill="var(--ink)"
        transform="translate(2.5, 2.5)"
      />
      <path
        d="M4 13h14V5l12 11-12 11v-8H4z"
        fill="var(--paper)"
        stroke="var(--ink)"
        strokeWidth={3}
        strokeLinejoin="round"
      />
    </svg>
  );
}
