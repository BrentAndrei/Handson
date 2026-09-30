import { cn } from "../../lib/cn";

type Surface =
  | "raised"
  | "sunken"
  | "teal"
  | "coral"
  | "lime"
  | "sun"
  | "purple"
  | "pink"
  | "ink";
type Depth = "sm" | "md" | "lg" | "xl";
type Tag = "div" | "section" | "article" | "li";

/** Props shared by every tag, minus the element-specific ones. */
type BaseProps = {
  surface?: Surface;
  depth?: Depth;
  as?: Tag;
  className?: string;
  children?: React.ReactNode;
  'aria-labelledby'?: string;
  'aria-label'?: string;
};

export type BrutalCardProps = BaseProps &
  Omit<React.HTMLAttributes<HTMLElement>, keyof BaseProps>;

const SURFACE: Record<Surface, string> = {
  raised: "bg-[var(--paper-raised)]",
  sunken: "bg-[var(--paper-sunken)]",
  teal: "bg-[var(--teal)] text-[var(--ink)]",
  coral: "bg-[var(--coral)] text-[var(--ink)]",
  lime: "bg-[var(--lime)] text-[var(--ink)]",
  sun: "bg-[var(--sun)] text-[var(--ink)]",
  /* Purple is a DECORATIVE fill only — it carries no text, because neither
     ink nor white clears 4.5:1 on it. Use "purple-deep" for text. */
  purple: "bg-[var(--purple)]",
  pink: "bg-[var(--pink)] text-[var(--ink)]",
  ink: "bg-[var(--ink)] text-[var(--paper)]",
};

const DEPTH: Record<Depth, string> = {
  sm: "shadow-[var(--shadow-sm)]",
  md: "shadow-[var(--shadow)]",
  lg: "shadow-[var(--shadow-lg)]",
  xl: "shadow-[var(--shadow-xl)]",
};

/**
 * A physical UI object: flat fill, 3px ink border, hard offset shadow.
 * Never uses blur, transparency, or a large radius.
 *
 * `as` is typed loosely (React.HTMLAttributes<HTMLElement>) because a
 * discriminated union across div/section/li would surface conflicting DOM
 * handler types without buying anything at the call site.
 */
export function BrutalCard({
  surface = "raised",
  depth = "md",
  as: Tag = "div",
  className,
  children,
  ...rest
}: BrutalCardProps) {
  return (
    <Tag
      className={cn(
        "rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)]",
        SURFACE[surface],
        DEPTH[depth],
        className
      )}
      {...rest}
    >
      {children}
    </Tag>
  );
}