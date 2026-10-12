import { useState, type FormEvent } from "react";
import { BrutalButton } from "../components/ui/BrutalButton";
import { BrutalCard } from "../components/ui/BrutalCard";
import { BrutalInput } from "../components/ui/BrutalInput";
import { Wordmark } from "./HomeScreen";
import { useAuth } from "../auth/AuthProvider";
import { IconSpark } from "../components/icons";
import type { AppView } from "../components/ui/BrutalNav";

type Mode = "signin" | "signup" | "forgot";

const COPY: Record<Mode, { kicker: string; title: string; cta: string }> = {
  signin: { kicker: "Welcome back", title: "Log in", cta: "Log in" },
  signup: { kicker: "Get started", title: "Create account", cta: "Create account" },
  forgot: { kicker: "Account recovery", title: "Reset password", cta: "Send reset link" },
};

/**
 * Client-side checks that run BEFORE any network call.
 *
 * Cheap format problems (a typo'd address, mismatched passwords) are answered
 * instantly and locally; anything that genuinely requires the server goes to
 * Firebase. `noValidate` on the form is deliberate — the browser's own bubble
 * would pre-empt these messages and we would never control the wording.
 */
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MIN_PASSWORD = 6;

function validate(mode: Mode, email: string, password: string, confirm: string): string | null {
  if (mode !== "signup" && mode !== "signin" && !EMAIL_RE.test(email.trim())) {
    return "Please enter a valid email.";
  }
  if (mode === "signup") {
    if (!EMAIL_RE.test(email.trim())) return "Please enter a valid email.";
    if (password.length < MIN_PASSWORD) {
      return `Password must be at least ${MIN_PASSWORD} characters.`;
    }
    if (password !== confirm) return "Passwords do not match.";
  }
  return null;
}

/**
 * One screen, three modes. Keeping them together preserves context when a user
 * moves between them — no separate routes to lose.
 */
export function LoginScreen({ onNavigate }: { onNavigate: (v: AppView) => void }) {
  const { signIn, signUp, requestPasswordReset, service, configured } = useAuth();
  const [mode, setMode] = useState<Mode>("signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const copy = COPY[mode];

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setNotice(null);

    const invalid = validate(mode, email, password, confirm);
    if (invalid) {
      setError(invalid);
      return;
    }

    setBusy(true);
    try {
      if (mode === "forgot") {
        await requestPasswordReset(email);
        // Firebase returns success whether or not the address exists; saying so
        // here avoids turning the reset form into an account-enumeration probe.
        setNotice("If that address has an account, a reset link is on its way. Check your inbox.");
      } else if (mode === "signup") {
        await signUp(name, email, password);
        // No navigation here: AuthProvider's listener re-renders the app once
        // Firebase reports the new user. Calling onNavigate would race it.
      } else {
        await signIn(email, password);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to connect. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const switchTo = (m: Mode) => {
    setMode(m);
    setError(null);
    setNotice(null);
  };

  return (
    <div className="mx-auto w-full max-w-md px-4 pb-8 pt-8 sm:px-6">
      <header className="mb-6 flex justify-center">
        <Wordmark className="text-xl tracking-tight" />
      </header>

      <BrutalCard depth="lg" className="p-5 sm:p-6">
        <p className="m-0 text-[11px] font-extrabold uppercase tracking-[0.22em] text-[var(--teal-deep)]">
          {copy.kicker}
        </p>
        <h1 className="m-0 mt-1 font-[family-name:var(--font-display)] text-3xl uppercase tracking-tight">
          {copy.title}
        </h1>

        <form onSubmit={onSubmit} className="mt-5 flex flex-col gap-4" noValidate>
          {mode === "signup" ? (
            <BrutalInput
              label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoComplete="name"
              placeholder="Your name"
            />
          ) : null}

          <BrutalInput
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            placeholder="you@example.com"
            required
          />

          {mode !== "forgot" ? (
            <BrutalInput
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              placeholder={mode === "signup" ? `At least ${MIN_PASSWORD} characters` : "••••••••"}
              hint={mode === "signup" ? `Use at least ${MIN_PASSWORD} characters.` : undefined}
              required
            />
          ) : null}

          {mode === "signup" ? (
            <BrutalInput
              label="Confirm password"
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              autoComplete="new-password"
              placeholder="Repeat your password"
              required
            />
          ) : null}

          {/* Errors and notices are announced, and never rely on colour alone. */}
          <div aria-live="polite">
            {error ? (
              <p className="m-0 rounded-[var(--radius)] border-[var(--border-w-thin)] border-[var(--danger)] bg-[var(--danger)]/10 px-3 py-2 text-sm font-bold text-[var(--danger)]">
                {error}
              </p>
            ) : null}
            {notice ? (
              <p className="m-0 rounded-[var(--radius)] border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--lime)] px-3 py-2 text-sm font-bold text-[var(--ink)]">
                {notice}
              </p>
            ) : null}
          </div>

          <BrutalButton type="submit" tone="teal" size="lg" block withArrow disabled={busy}>
            {busy ? "Working…" : copy.cta}
          </BrutalButton>
        </form>
        <div className="mt-5 flex flex-col gap-2 border-t-[var(--border-w-thin)] border-[var(--ink)] pt-4 text-sm">
          {mode !== "signin" ? (
            <button type="button" onClick={() => switchTo("signin")} className="self-start font-extrabold text-[var(--teal-deep)] underline underline-offset-4">
              Already have an account? Log in
            </button>
          ) : null}
          {mode !== "signup" ? (
            <button type="button" onClick={() => switchTo("signup")} className="self-start font-extrabold text-[var(--teal-deep)] underline underline-offset-4">
              Create an account
            </button>
          ) : null}
          {mode !== "forgot" ? (
            <button type="button" onClick={() => switchTo("forgot")} className="self-start font-extrabold text-[var(--ink-soft)] underline underline-offset-4">
              Forgot password?
            </button>
          ) : null}
        </div>
      </BrutalCard>

      {/* Setup notice, shown only when the Firebase env vars are missing. */}
      {!configured ? (
        <p className="m-4 flex items-start gap-2 rounded-[var(--radius)] border-[var(--border-w-thin)] border-[var(--danger)] bg-[var(--danger)]/10 px-3 py-2 text-xs leading-relaxed text-[var(--danger)]">
          <IconSpark className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>
            <strong className="font-extrabold">Firebase not configured.</strong> Copy
            <code className="mx-1 font-mono">.env.example</code> to
            <code className="mx-1 font-mono">.env</code>, paste in your Firebase web config,
            then restart the dev server.
          </span>
        </p>
      ) : null}

      {/* What this account actually is now that there is a real backend. */}
      {configured && !service.isSecureBackend ? (
        <p className="m-4 flex items-start gap-2 rounded-[var(--radius)] border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-3 py-2 text-xs leading-relaxed text-[var(--ink-soft)]">
          <IconSpark className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>
            <strong className="font-extrabold">Local profile only.</strong> This build is not
            connected to a real auth provider.
          </span>
        </p>
      ) : null}

      <div className="mt-4 text-center">
        <button type="button" onClick={() => onNavigate("home")} className="font-extrabold text-[var(--ink-soft)] underline underline-offset-4">
          Back to home
        </button>
      </div>
    </div>
  );
}
