/**
 * HANDSON AUTH
 * ---------------------------------------------------------------------------
 * There is NO backend in this project. Rather than fake a secure login, the
 * three concerns are split so a real provider can be dropped in without
 * touching any screen:
 *
 *   AuthService  — the transport. LocalStorage today; swap for Supabase/Firebase
 *                  by reimplementing this one interface.
 *   AuthContext  — user state + subscribe, for React.
 *   <screens>    — only ever talk to AuthContext, never to the transport.
 *
 * SECURITY: `LocalAuthService` stores a password-derived digest in
 * localStorage. It is NOT authentication — it is a local profile. Anyone with
 * the device and devtools can read it. The Account screen says so plainly
 * rather than implying protection that does not exist.
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
  /** Whether this backend offers real, server-verified auth. */
  readonly isSecureBackend: boolean;
}

const STORAGE_KEY = "handson.auth.v1";
const USERS_KEY = "handson.auth.users.v1";

const normalise = (email: string) => email.trim().toLowerCase();

/**
 * Deterministic, dependency-free digest. Deliberately NOT a security primitive —
 * it exists so a stored value is not a plaintext password in devtools.
 */
function digest(input: string): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < input.length; i++) {
    const c = input.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul(h2 + c, 0x85ebca6b) >>> 0;
  }
  return (h1 >>> 0).toString(16).padStart(8, "0") + (h2 >>> 0).toString(16).padStart(8, "0");
}

interface StoredUser extends AuthUser {
  secret: string;
}

function readUsers(): StoredUser[] {
  try {
    const raw = localStorage.getItem(USERS_KEY);
    return raw ? (JSON.parse(raw) as StoredUser[]) : [];
  } catch {
    return [];
  }
}

function writeUsers(list: StoredUser[]): void {
  try {
    localStorage.setItem(USERS_KEY, JSON.stringify(list));
  } catch {
    /* storage unavailable (private mode) — the session simply will not persist */
  }
}

const toPublic = (u: StoredUser): AuthUser => ({
  id: u.id,
  name: u.name,
  email: u.email,
  createdAt: u.createdAt,
});

/* -------------------------------------------------------------------------- */

export class LocalAuthService implements AuthService {
  readonly isSecureBackend = false;

  async signUp(name: string, email: string, password: string): Promise<AuthResult> {
    const key = normalise(email);
    const users = readUsers();
    if (users.some((u) => u.email === key)) {
      throw new Error("An account with that email already exists on this device.");
    }
    if (password.length < 8) {
      throw new Error("Password must be at least 8 characters.");
    }
    const user: StoredUser = {
      id: `u_${Date.now().toString(36)}`,
      name: name.trim() || key.split("@")[0],
      email: key,
      createdAt: Date.now(),
      secret: digest(`${key}:${password}`),
    };
    users.push(user);
    writeUsers(users);
    this.persist(user);
    return { user: toPublic(user) };
  }

  async signIn(email: string, password: string): Promise<AuthResult> {
    const key = normalise(email);
    const user = readUsers().find((u) => u.email === key);
    if (!user || user.secret !== digest(`${key}:${password}`)) {
      throw new Error("Email or password is incorrect.");
    }
    this.persist(user);
    return { user: toPublic(user) };
  }

  async signOut(): Promise<void> {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* nothing to remove */
    }
  }

  async restore(): Promise<AuthUser | null> {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const { id, email } = JSON.parse(raw) as { id: string; email: string };
      const user = readUsers().find((u) => u.id === id && u.email === email);
      return user ? toPublic(user) : null;
    } catch {
      return null;
    }
  }

  async requestPasswordReset(_email: string): Promise<void> {
    throw new Error(
      "Password reset needs a backend. On this build accounts live only in this browser — clear site data to start fresh."
    );
  }

  private persist(user: StoredUser): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ id: user.id, email: user.email }));
    } catch {
      /* non-fatal: the session just will not survive a reload */
    }
  }
}

/* -------------------------------------------------------------------------- */

interface AuthContextValue {
  user: AuthUser | null;
  status: "loading" | "ready";
  service: AuthService;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  requestPasswordReset: (email: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/**
 * Defaults to the local service so the app works before any provider is wired.
 * Pass a real `service` and every screen is unchanged.
 */
export function AuthProvider({
  children,
  service = new LocalAuthService(),
}: {
  children: ReactNode;
  service?: AuthService;
}) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<AuthContextValue["status"]>("loading");

  useEffect(() => {
    let alive = true;
    service
      .restore()
      .then((u) => {
        if (!alive) return;
        setUser(u);
        setStatus("ready");
      })
      .catch(() => {
        if (!alive) return;
        setStatus("ready");
      });
    return () => {
      alive = false;
    };
  }, [service]);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const { user: u } = await service.signIn(email, password);
      setUser(u);
    },
    [service]
  );

  const signUp = useCallback(
    async (name: string, email: string, password: string) => {
      const { user: u } = await service.signUp(name, email, password);
      setUser(u);
    },
    [service]
  );

  const signOut = useCallback(async () => {
    await service.signOut();
    setUser(null);
  }, [service]);

  const requestPasswordReset = useCallback(
    async (email: string) => service.requestPasswordReset(email),
    [service]
  );

  const value = useMemo<AuthContextValue>(
    () => ({ user, status, service, signIn, signUp, signOut, requestPasswordReset }),
    [user, status, service, signIn, signUp, signOut, requestPasswordReset]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/** Screens consume this, never the transport directly. */
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>.");
  return ctx;
}
