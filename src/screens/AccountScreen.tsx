import { BrutalButton } from "../components/ui/BrutalButton";
import { BrutalCard } from "../components/ui/BrutalCard";
import { BrutalBadge } from "../components/ui/BrutalBadge";
import { IconLogout, IconUser, IconCamera, IconCubeHand } from "../components/icons";
import { useAuth } from "../auth/AuthProvider";
import type { AppView } from "../components/ui/BrutalNav";

/** Reads the account's creation date without pulling in a date library. */
function formatDate(ts: number): string {
  try {
    return new Date(ts).toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  } catch {
    return "Unknown";
  }
}

export function AccountScreen({
  onNavigate,
  onSignInRequested,
}: {
  onNavigate: (v: AppView) => void;
  onSignInRequested: () => void;
}) {
  const { user, status, signOut, service } = useAuth();

  if (status === "loading") {
    return (
      <div className="mx-auto w-full max-w-md px-4 py-16 text-center">
        <p className="font-bold text-[var(--ink-soft)]" role="status">
          Loading your profile…
        </p>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="mx-auto w-full max-w-md px-4 pb-8 pt-10 sm:px-6">
        <BrutalCard depth="lg" className="p-6 text-center">
          <IconUser className="mx-auto h-16 w-16 text-[var(--ink-soft)]" aria-hidden />
          <h1 className="m-0 mt-3 font-[family-name:var(--font-display)] text-2xl uppercase tracking-tight">
            No account yet
          </h1>
          <p className="m-0 mt-2 text-[15px] leading-relaxed text-[var(--ink-soft)]">
            Create a profile to keep your name across visits and sign out on shared devices.
          </p>
          <div className="mt-5">
            <BrutalButton
              tone="teal"
              size="lg"
              block
              withArrow
              onClick={() => onSignInRequested()}
            >
              Log in or sign up
            </BrutalButton>
          </div>
        </BrutalCard>
      </div>
    );
  }

  const initials = user.name.trim().slice(0, 2).toUpperCase() || "HS";

  return (
    <div className="mx-auto w-full max-w-md px-4 pb-8 pt-6 sm:px-6">
      <h1 className="m-0 font-[family-name:var(--font-display)] text-3xl uppercase tracking-tight">
        Account
      </h1>

      {/* Identity block */}
      <BrutalCard depth="lg" className="mt-4 p-5">
        <div className="flex items-center gap-4">
          <span
            className="flex h-16 w-16 shrink-0 items-center justify-center rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)] bg-[var(--lime)] font-[family-name:var(--font-display)] text-xl"
            aria-hidden="true"
          >
            {initials}
          </span>
          <div className="min-w-0">
            <p className="m-0 truncate font-[family-name:var(--font-display)] text-lg uppercase tracking-tight">
              {user.name}
            </p>
            <p className="m-0 truncate text-sm text-[var(--ink-soft)]">{user.email}</p>
            <p className="m-0 mt-1">
              <BrutalBadge tone="paper">Since {formatDate(user.createdAt)}</BrutalBadge>
            </p>
          </div>
        </div>

        <div className="mt-5">
          <BrutalButton tone="ink" size="md" block onClick={() => void signOut()}>
            <IconLogout className="h-4 w-4" aria-hidden />
            Log out
          </BrutalButton>
        </div>
      </BrutalCard>

      {/* What this profile is and is not. */}
      <BrutalCard surface="sunken" depth="sm" className="mt-4 p-4">
        <h2 className="m-0 text-xs font-extrabold uppercase tracking-[0.16em] text-[var(--ink-soft)]">
          About this profile
        </h2>
        <p className="m-0 mt-2 text-sm leading-relaxed text-[var(--ink-soft)]">
          {service.isSecureBackend
            ? "Your account is managed by the HandSon authentication service."
            : "Stored in this browser only. There is no HandSon server, so this is a local profile — not real authentication. Clearing site data deletes it, and the password is not recoverable."}
        </p>
      </BrutalCard>

      {/* Quick jumps back into the two real features. */}
      <div className="mt-4 grid gap-3">
        <BrutalButton tone="coral" size="md" block onClick={() => onNavigate("translate")}>
          <IconCamera className="h-4 w-4" aria-hidden />
          Real-time translator
        </BrutalButton>
        <BrutalButton tone="teal" size="md" block onClick={() => onNavigate("avatar")}>
          <IconCubeHand className="h-4 w-4" aria-hidden />
          3D sign avatar
        </BrutalButton>
      </div>
    </div>
  );
}