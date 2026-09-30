import { cn } from "../../lib/cn";

export type StickerTone =
  | "ink"
  | "paper"
  | "teal"
  | "coral"
  | "lime"
  | "sun"
  | "pink"
  | "purple-deep";

/** ink-on-bright and white-on-dark are the only two approved text pairings. */
const TONE: Record<StickerTone, string> = {
  ink: "bg-[var(--ink)] text-[var(--paper)]",
  paper: "bg-[var(--paper-raised)] text-[var(--ink)]",
  teal: "bg-[var(--teal)] text-[var(--ink)]",
  coral: "bg-[var(--coral)] text-[var(--ink)]",
  lime: "bg-[var(--lime)] text-[var(--ink)]",
  sun: "bg-[var(--sun)] text-[var(--ink)]",
  pink: "bg-[var(--pink)] text-[var(--ink)]",
  "purple-deep": "bg-[var(--purple-deep)] text-white",
};

export type StickerRotate = "none" | "-3" | "-2" | "-1" | "1" | "2" | "3" | "4";

const ROTATE: Record<StickerRotate, string> = {
  none: "",
  "-3": "hs-rot-neg3",
  "-2": "hs-rot-neg2",
  "-1": "hs-rot-neg1",
  "1": "hs-rot-1",
  "2": "hs-rot-2",
  "3": "hs-rot-3",
  "4": "hs-rot-4",
};

export interface BrutalStickerProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: StickerTone;
  /**
   * Rotation in degrees. Kept small and explicit: a sticker at 4deg looks
   * placed by hand, a card at 4deg looks broken.
   */
  rotate?: StickerRotate;
  /** Sits above neighbours when it overlaps them. */
  raised?: boolean;
}

/**
 * A printed label that overlaps its neighbours.
 *
 * Stickers are how this design creates depth without adding layout: a label is
 * pulled out of flow with a negative margin and given a z-index, so it reads as
 * a physical thing laid on top of the card rather than a row of chips inside it.
 */
export function BrutalSticker({
  tone = "paper",
  rotate = "none",
  raised = false,
  className,
  children,
  ...rest
}: BrutalStickerProps) {
  return (
    <span
      className={cn(
        "hs-sticker",
        ROTATE[rotate],
        TONE[tone],
        raised && "z-20",
        className
      )}
      {...rest}
    >
      {children}
    </span>
  );
}
