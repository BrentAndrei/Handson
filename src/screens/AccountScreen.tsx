import { BrutalButton } from "../components/ui/BrutalButton";
import { BrutalCard } from "../components/ui/BrutalCard";
import { BrutalBadge } from "../components/ui/BrutalBadge";
import { IconLogout, IconCamera, IconCubeHand, IconSpark } from "../components/icons";
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

/**
 * The signed-in profile.
 *
 * Everything shown here comes from the Firebase user object — there is no
 * placeholder account. The unauthenticated case is not handled here at all:
 * the gate in App.tsx means this screen only mounts when `user` exists.
 */
export function AccountScreen({ onNavigate, onReplayTour }: { onNavigate: (v: AppView) => void; onReplayTour?: () => void }) {
  const { user, signOut, service } = useAuth();

  if (!user) {
    // Transient only: a sign-out can land here between this render and the
    // gate swapping the tree over. Never show invented identity details.
    return (
      <div className="mx-auto w-full max-w-md px-4 py-16 text-center">
        <p className="font-bold text-[var(--ink-soft)]" role="status">
          Loading your profile…
        </p>
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
            ? "Signed in with Firebase Authentication. Your account and password are managed by Google — HandSon never stores your password."
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
        {onReplayTour ? (
          <BrutalButton tone="paper" size="md" block onClick={onReplayTour}>
            <IconSpark className="h-4 w-4" aria-hidden />
            Replay the tour
          </BrutalButton>
        ) : null}
      </div>
    </div>
  );
}