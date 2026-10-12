/**
 * HANDSON AUTH
 * ---------------------------------------------------------------------------
 * Firebase Authentication is the single source of truth for "who is signed in".
 * The three concerns stay split so no screen ever talks to the SDK directly:
 *
 *   AuthService     — the transport. FirebaseAuthService, below.
 *   AuthContext     — user state + subscribe, for React.
 *   <screens>       — only ever talk to AuthContext, never to the transport.
 *
 * Firebase owns credentials, sessions and password storage. HandSon never
 * writes a password, never keeps its own session token, and has no local
 * "logged in" flag: `onAuthStateChanged` is the only thing that decides
 * whether the app renders.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
  signOut as fbSignOut,
  updateProfile,
  type User,
} from "firebase/auth";
import { getFirebaseAuth, isFirebaseConfigured } from "../lib/firebase";

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  createdAt: number;
}

export interface AuthResult {
  user: AuthUser;
}

export interface AuthService {
  signIn(email: string, password: string): Promise<AuthResult>;
  signUp(name: string, email: string, password: string): Promise<AuthResult>;
  signOut(): Promise<void>;
  /** Resolves to the persisted user, or null. */
  restore(): Promise<AuthUser | null>;
  requestPasswordReset(email: string): Promise<void>;
  /**
   * Streams auth state. Returns an unsubscribe function. This is what makes
   * sign-out, cross-tab sign-in and session restore all drive the same UI.
   */
  subscribe(listener: (user: AuthUser | null) => void): () => void;
  /** Whether this backend offers real, server-verified auth. */
  readonly isSecureBackend: boolean;
}

const normalise = (email: string) => email.trim().toLowerCase();

/** Maps a Firebase user onto the shape the screens consume. */
const toPublic = (u: User): AuthUser => {
  const createdAt = u.metadata?.creationTime ? Date.parse(u.metadata.creationTime) : Date.now();
  return {
    id: u.uid,
    // Firebase email/password accounts have no display name by default; fall
    // back to the local part of the address so the UI always has something.
    name: u.displayName?.trim() || (u.email ? u.email.split("@")[0] : "HandSon user"),
    email: u.email ?? "",
    createdAt: Number.isNaN(createdAt) ? Date.now() : createdAt,
  };
};

/**
 * Turns a Firebase error code into something a person can act on.
 *
 * Without this the raw SDK message reaches the UI — things like
 * "Firebase: Error (auth/invalid-credential)." That leaks implementation
 * detail and tells a user nothing about what to change.
 */
export function friendlyAuthError(err: unknown): string {
  const code =
    typeof err === "object" && err !== null && "code" in err
      ? String((err as { code: unknown }).code)
      : "";

  switch (code) {
    case "auth/invalid-email":
      return "Please enter a valid email.";
    case "auth/missing-password":
    case "auth/empty-password":
      return "Please enter your password.";
    case "auth/invalid-credential":
    case "auth/wrong-password":
    case "auth/user-not-found":
      // Deliberately identical: saying which half was wrong leaks whether an
      // account exists.
      return "Incorrect email or password.";
    case "auth/email-already-in-use":
      return "An account with that email already exists.";
    case "auth/weak-password":
      return "Password is too weak. Use at least 6 characters.";
    case "auth/too-many-requests":
      return "Too many attempts. Please wait a moment and try again.";
    case "auth/network-request-failed":
    case "auth/internal-error":
    case "auth/unauthorized-domain":
      return "Unable to connect. Please try again.";
    case "auth/operation-not-allowed":
      return "Email and password sign-in is not enabled for this project.";
    case "auth/requires-recent-login":
      return "Please log in again before making this change.";
    default:
      return "Unable to connect. Please try again.";
  }
}

/* -------------------------------------------------------------------------- */

export class FirebaseAuthService implements AuthService {
  readonly isSecureBackend = true;

  private auth() {
    if (!isFirebaseConfigured()) {
      return Promise.reject(
        new Error(
          "Firebase is not configured. Copy .env.example to .env and add your Firebase web config, then restart the dev server."
        )
      );
    }
    return getFirebaseAuth();
  }

  async signIn(email: string, password: string): Promise<AuthResult> {
    try {
      const auth = await this.auth();
      const cred = await signInWithEmailAndPassword(auth, normalise(email), password);
      return { user: toPublic(cred.user) };
    } catch (err) {
      throw new Error(friendlyAuthError(err));
    }
  }

  async signUp(name: string, email: string, password: string): Promise<AuthResult> {
    try {
      const auth = await this.auth();
      const cred = await createUserWithEmailAndPassword(auth, normalise(email), password);
      // displayName is a separate call after creation; failing it must not fail
      // the signup, since the account already exists at that point.
      const trimmed = name.trim();
      if (trimmed) {
        await updateProfile(cred.user, { displayName: trimmed }).catch(() => {});
      }
      return { user: toPublic(cred.user) };
    } catch (err) {
      throw new Error(friendlyAuthError(err));
    }
  }

  async signOut(): Promise<void> {
    try {
      const auth = await this.auth();
      await fbSignOut(auth);
    } catch (err) {
      throw new Error(friendlyAuthError(err));
    }
  }

  async restore(): Promise<AuthUser | null> {
    try {
      const auth = await this.auth();
      return auth.currentUser ? toPublic(auth.currentUser) : null;
    } catch {
      return null;
    }
  }

  async requestPasswordReset(email: string): Promise<void> {
    try {
      const auth = await this.auth();
      await sendPasswordResetEmail(auth, normalise(email));
    } catch (err) {
      throw new Error(friendlyAuthError(err));
    }
  }

  subscribe(listener: (user: AuthUser | null) => void): () => void {
    let unsubscribe: (() => void) | null = null;
    let cancelled = false;

    // An unconfigured project must still resolve to "signed out", not throw
    // during render — otherwise the app renders a blank page instead of the
    // login screen with its setup notice.
    if (!isFirebaseConfigured()) {
      listener(null);
      return () => {};
    }

    getFirebaseAuth()
      .then((auth) => {
        if (cancelled) return;
        unsubscribe = onAuthStateChanged(auth, (u) => listener(u ? toPublic(u) : null));
      })
      .catch(() => listener(null));

    return () => {
      cancelled = true;
      unsubscribe?.();
    };
  }
}

/* -------------------------------------------------------------------------- */

interface AuthContextValue {
  user: AuthUser | null;
  /**
   * `loading` until Firebase has resolved the persisted session. The app MUST
   * not render before this flips — otherwise a signed-in user sees the login
   * screen flash on every reload.
   */
  status: "loading" | "authenticated" | "unauthenticated";
  service: AuthService;
  /** False when the VITE_FIREBASE_* env vars are missing. */
  configured: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  requestPasswordReset: (email: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/**
 * Owns auth state for the whole tree.
 *
 * Firebase's `onAuthStateChanged` is the ONLY writer of `user`. signIn/signUp
 * do not optimistically set it, and signOut does not clear it by hand — the
 * SDK emits the change and the listener applies it. That is what keeps the UI
 * honest: what is on screen is always what Firebase believes.
 */
export function AuthProvider({
  children,
  service,
}: {
  children: ReactNode;
  service?: AuthService;
}) {
  // MUST be memoised. A `service = new FirebaseAuthService()` default parameter
  // would build a new instance on every render, and because the subscribe
  // effect depends on `service` it would then unsubscribe and resubscribe on
  // every render — churning auth state and dropping the session.
  const [fallbackService] = useState(() => new FirebaseAuthService());
  const activeService = service ?? fallbackService;

  const [user, setUser] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<AuthContextValue["status"]>("loading");
  const configured = isFirebaseConfigured();

  useEffect(() => {
    setStatus("loading");
    // `subscribe` resolves Firebase's initial state before its first callback,
    // so "loading" genuinely means "Firebase has not answered yet".
    const unsubscribe = activeService.subscribe((u) => {
      setUser(u);
      setStatus(u ? "authenticated" : "unauthenticated");
    });
    return unsubscribe;
  }, [activeService]);

  // These delegate to Firebase and return; the listener above flips the UI.
  const signIn = useCallback(
    async (email: string, password: string) => {
      await activeService.signIn(email, password);
    },
    [activeService]
  );

  const signUp = useCallback(
    async (name: string, email: string, password: string) => {
      await activeService.signUp(name, email, password);
    },
    [activeService]
  );

  const signOut = useCallback(async () => {
    await activeService.signOut();
  }, [activeService]);

  const requestPasswordReset = useCallback(
    async (email: string) => activeService.requestPasswordReset(email),
    [activeService]
  );

  const value = useMemo<AuthContextValue>(
    () => ({ user, status, service: activeService, configured, signIn, signUp, signOut, requestPasswordReset }),
    [user, status, activeService, configured, signIn, signUp, signOut, requestPasswordReset]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/** Screens consume this, never the transport directly. */
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>.");
  return ctx;
}