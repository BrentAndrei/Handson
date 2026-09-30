import { cn } from "../../lib/cn";
import { IconCamera, IconHome, IconUser, IconAvatar, type IconProps } from "../icons";

export type AppView = "home" | "translate" | "avatar" | "account";

export interface NavItem {
  id: AppView;
  label: string;
  Icon: (p: IconProps) => JSX.Element;
  /** Fill colour of the active block. Roles come from the palette. */
  activeTone: "sun" | "coral" | "lime" | "pink";
  /** Printed index, part of the editorial numbering. */
  index: string;
}

export const NAV_ITEMS: NavItem[] = [
  { id: "home", label: "Home", Icon: IconHome, activeTone: "sun", index: "01" },
  { id: "translate", label: "Translate", Icon: IconCamera, activeTone: "coral", index: "02" },
  { id: "avatar", label: "Avatar", Icon: IconAvatar, activeTone: "lime", index: "03" },
  { id: "account", label: "Account", Icon: IconUser, activeTone: "pink", index: "04" },
];

const ACTIVE_FILL: Record<NavItem["activeTone"], string> = {
  sun: "bg-[var(--sun)] text-[var(--ink)]",
  coral: "bg-[var(--coral)] text-[var(--ink)]",
  lime: "bg-[var(--lime)] text-[var(--ink)]",
  pink: "bg-[var(--pink)] text-[var(--ink)]",
};

export interface BrutalNavProps {
  active: AppView;
  onNavigate: (v: AppView) => void;
}

/**
 * A bordered horizontal navigation SLAB pinned to the bottom of the viewport.
 *
 * Why a slab and not a tab bar: a tab bar is a phone idiom that reads as
 * generic. A slab is a printed object — thick top rule, hard shadow above it,
 * and each item is a discrete printed block that can carry a colour.
 *
 * FIXED, not sticky: the nav is a child of <main>, which is a max-width column,
 * so `sticky bottom-0` would pin it to the bottom of THAT column and leave it
 * floating mid-viewport.
 *
 * ACTIVE STATE IS NEVER COLOUR-ONLY: the active item gains a filled block AND
 * a visible border AND `aria-current="page"`. Each item is a real <button>
 * inside an <li>, so it is tab-navigable and its accessible name is exactly
 * the visible label.
 */
export function BrutalNav({ active, onNavigate }: BrutalNavProps) {
  return (
    <nav
      aria-label="Primary"
      className="fixed inset-x-0 bottom-0 z-40 border-t-[var(--border-w-thick)] border-[var(--ink)] bg-[var(--paper-raised)] shadow-[0_-6px_0_var(--ink)]"
    >
      <ul className="mx-auto flex max-w-3xl items-stretch gap-1.5 px-2 py-2 sm:gap-2 sm:px-3">
        {NAV_ITEMS.map(({ id, label, Icon, activeTone, index }) => {
          const isActive = id === active;
          return (
            <li key={id} className="flex-1">
              <button
                type="button"
                onClick={() => onNavigate(id)}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "relative flex min-h-[52px] w-full flex-col items-center justify-center gap-0.5",
                  "rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)]",
                  "font-[family-name:var(--font-display)] text-[10px] uppercase leading-none tracking-[0.08em]",
                  "transition-[transform,box-shadow,background-color,color] duration-[var(--press-ms)] ease-[var(--ease-snap)]",
                  "hover:-translate-y-[2px] hover:shadow-[var(--shadow-sm)]",
                  "active:translate-x-[2px] active:translate-y-[2px] active:shadow-[var(--shadow-pressed)]",
                  "motion-reduce:transition-none motion-reduce:hover:translate-y-0",
                  isActive
                    ? cn(
                        "-translate-y-[3px] shadow-[var(--shadow-sm)]",
                        ACTIVE_FILL[activeTone]
                      )
                    : "bg-transparent text-[var(--ink-soft)] hover:bg-[var(--paper-sunken)]"
                )}
              >
                {/* Printed index. Rendered via CSS `content` on a pseudo-element
                    rather than as a text node, so it stays out of the button's
                    textContent entirely. The gate (and assistive tech) read
                    textContent, so an aria-hidden child would still have been
                    picked up by the former. */}
                <span aria-hidden="true" className="hs-nav-index" data-index={index} />
                <Icon className="mt-1 h-[22px] w-[22px]" aria-hidden />
                <span>{label}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}