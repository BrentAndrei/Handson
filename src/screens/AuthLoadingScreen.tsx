import { Wordmark } from "./HomeScreen";

/**
 * Shown while Firebase resolves the persisted session.
 *
 * This exists so the app never flashes the home screen at an unauthenticated
 * visitor. Rendering nothing (or the login form) before `onAuthStateChanged`
 * answers would let protected UI appear for a frame, and would also show the
 * login screen to a signed-in user on every reload.
 */
export function AuthLoadingScreen() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-5 px-6 text-center">
      <Wordmark className="text-2xl tracking-tight" />
      <p
        className="font-bold text-[var(--ink-soft)]"
        role="status"
        aria-live="polite"
      >
        Checking your session…
      </p>
    </div>
  );
}
