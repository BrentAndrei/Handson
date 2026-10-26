import { ReactNode } from "react";
import { BrutalButton } from "../components/ui/BrutalButton";
import { BrutalLabel } from "../components/ui/BrutalLabel";
import { IconSpark } from "../components/icons";

interface PrintedLayoutProps {
  children: ReactNode;
  /** Eyebrow kicker above the page title, e.g. "Real-time translator" */
  kicker?: string;
  /** Page heading */
  title: string;
  /** Optional subtitle under the heading */
  subtitle?: string;
  /** Whether to show a "Back to home" button */
  showBack?: boolean;
  onBack?: () => void;
  /** Extra className on the outer wrapper */
  className?: string;
  /** Section rule label between the header and children (default: none) */
  ruleLabel?: string;
}

/**
 * Shared printed-page shell.
 *
 * Carries the printed backdrop (sun wash, halftone, sparks, registration
 * marks) that HomeScreen and VideoCaptureScreen each re-implemented, plus a
 * consistent header block (kicker + title + optional back button) and an
 * editorial section rule.
 *
 * Every screen that uses this gets the same visual language by construction
 * — there is no way to opt out of the paper/ink tokens, which is the point.
 */
export function PrintedLayout({
  children,
  kicker,
  title,
  subtitle,
  showBack = true,
  onBack,
  className = "",
  ruleLabel,
}: PrintedLayoutProps) {
  return (
    <div className={`relative ${className}`}>
      {/* PRINTED BACKDROP — same visual language as the home screen.
          All pieces are aria-hidden, pointer-events-none, locked to -z-10. */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
        <span className="absolute -top-24 right-0 block h-[300px] w-[300px] rounded-full bg-[var(--sun)] sm:-top-28 sm:h-[360px] sm:w-[360px]" />
        <span className="hs-dots hs-rot-neg2 absolute -bottom-4 left-0 block h-[90px] w-[150px] opacity-30" />
        <IconSpark className="absolute left-[3px] top-[80px] h-7 w-7 text-[var(--coral)] hs-rot-3" />
        <IconSpark className="absolute right-[70px] top-[150px] h-5 w-5 text-[var(--teal)] hs-rot-neg3" />
        <IconSpark className="absolute bottom-[190px] right-[14px] h-6 w-6 text-[var(--lime)] hs-rot-1" />
      </div>

      {/* Registration marks, asymmetric on purpose. */}
      <span aria-hidden="true" className="hs-mark-cross absolute -right-1 top-16 hidden opacity-40 lg:block" />
      <span aria-hidden="true" className="hs-mark-cross absolute left-1/2 top-4 hidden opacity-30 lg:block" />

      {/* Header */}
      <div className="flex items-center gap-3">
        {showBack && (
          <BrutalButton tone="paper" size="sm" onClick={onBack}>
            ← Back
          </BrutalButton>
        )}
        <div>
          {kicker && (
            <p className="m-0 text-[11px] font-extrabold uppercase tracking-[0.22em] text-[var(--teal-deep)]">
              {kicker}
            </p>
          )}
          <h1 className="m-0 font-[family-name:var(--font-display)] text-3xl uppercase tracking-tight">
            {title}
          </h1>
          {subtitle && (
            <p className="m-0 mt-1 text-sm font-semibold text-[var(--ink-soft)]">
              {subtitle}
            </p>
          )}
        </div>
      </div>

      {/* Section rule with spark — editorial device tying header to content. */}
      {ruleLabel && (
        <div className="mb-5 mt-5 flex items-center gap-3">
          <BrutalLabel tone="mute">{ruleLabel}</BrutalLabel>
          <span aria-hidden="true" className="h-[3px] flex-1 bg-[var(--ink)]" />
          <IconSpark aria-hidden className="h-4 w-4 shrink-0 text-[var(--coral)] hs-rot-3" />
        </div>
      )}

      {/* Content */}
      <div className="mt-5">
        {children}
      </div>
    </div>
  );
}