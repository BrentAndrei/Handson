import { forwardRef, useId, useState } from "react";
import { cn } from "../../lib/cn";
import { IconEye, IconEyeOff } from "../icons";

export interface BrutalInputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> {
  label: string;
  /** Error text. Rendered in a danger-tinted block and wired via aria-describedby. */
  error?: string;
  hint?: string;
}

/**
 * A labelled text field.
 *
 * The label is a real <label> (not a placeholder) so the field stays
 * identifiable once it has content, and the error is programmatically
 * associated via aria-describedby + aria-invalid.
 */
export const BrutalInput = forwardRef<HTMLInputElement, BrutalInputProps>(
  function BrutalInput({ label, error, hint, className, id, ...rest }, ref) {
    const autoId = useId();
    const inputId = id ?? autoId;
    const msgId = `${inputId}-msg`;
    const [reveal, setReveal] = useState(false);
    const isPassword = rest.type === "password";

    return (
      <div className={cn("flex w-full flex-col gap-1.5", className)}>
        <label
          htmlFor={inputId}
          className="text-xs font-extrabold uppercase tracking-[0.14em] text-[var(--ink-soft)]"
        >
          {label}
        </label>

        <div className="relative">
          <input
            ref={ref}
            id={inputId}
            aria-invalid={error ? true : undefined}
            aria-describedby={error || hint ? msgId : undefined}
            className={cn(
              "min-h-[48px] w-full rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)]",
              "bg-[var(--paper-raised)] px-3 py-2 text-base text-[var(--ink)]",
              "placeholder:text-[var(--ink-mute)]",
              "shadow-[var(--shadow-sm)]",
              "focus-visible:outline-[var(--border-w)] focus-visible:outline-offset-2",
              error && "bg-[var(--danger)]/10",
              isPassword && "prong pr-12"
            )}
            {...rest}
            {...(isPassword ? { type: reveal ? "text" : "password" } : null)}
          />

          {isPassword ? (
            <button
              type="button"
              onClick={() => setReveal((v) => !v)}
              className={cn(
                "absolute right-2 top-1/2 flex h-9 w-9 -translate-y-1/2 items-center justify-center",
                "rounded-[var(--radius-sm)] border-[var(--border-w-thin)] border-[var(--ink)]",
                "bg-[var(--paper-sunken)] text-[var(--ink)]"
              )}
              aria-label={reveal ? "Hide password" : "Show password"}
              aria-pressed={reveal}
            >
              {reveal ? (
                <IconEyeOff className="h-5 w-5" aria-hidden />
              ) : (
                <IconEye className="h-5 w-5" aria-hidden />
              )}
            </button>
          ) : null}
        </div>

        {error || hint ? (
          <p
            id={msgId}
            className={cn(
              "m-0 text-sm font-semibold",
              error ? "text-[var(--danger)]" : "text-[var(--ink-mute)]"
            )}
          >
            {error ?? hint}
          </p>
        ) : null}
      </div>
    );
  }
);