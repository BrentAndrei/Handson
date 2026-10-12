/**
 * FIREBASE AUTH TEST BACKEND
 * ---------------------------------------------------------------------------
 * The app is gated behind Firebase Authentication, so the UI gate can no longer
 * reach the home screen without a signed-in user. Rather than weaken the gate
 * or stub it out in app code, this module answers the Firebase Web SDK's REST
 * calls directly in the browser:
 *
 *   POST identitytoolkit.googleapis.com/v1/accounts:signInWithPassword
 *   POST identitytoolkit.googleapis.com/v1/accounts:signUp
 *   POST identitytoolkit.googleapis.com/v1/accounts:lookup
 *   POST identitytoolkit.googleapis.com/v1/accounts:sendOobCode
 *   POST securetoken.googleapis.com/v1/token
 *
 * Everything else is passed straight through. The real SDK runs unmodified, so
 * this exercises the genuine sign-in path — only the server is local.
 *
 * The ID token below is an UNSIGNED JWT. That is safe here and nowhere else:
 * the browser SDK decodes the token's claims without verifying its signature
 * (signature verification is the server's job), and this token is never
 * accepted by any real service. It exists only to satisfy the decoder.
 */
import type { Page, HTTPRequest } from "puppeteer-core";

export const TEST_EMAIL = "tester@handson.test";
export const TEST_PASSWORD = "correct-horse-battery";
export const TEST_UID = "handson-test-uid-0001";
export const TEST_PROJECT = "handson-verify";

const b64url = (obj: unknown) =>
  Buffer.from(JSON.stringify(obj)).toString("base64url");

function makeIdToken(email: string): string {
  const now = Math.floor(Date.now() / 1000);
  return [
    b64url({ alg: "RS256", typ: "JWT", kid: "test-key" }),
    b64url({
      iss: `https://securetoken.google.com/${TEST_PROJECT}`,
      aud: TEST_PROJECT,
      sub: TEST_UID,
      iat: now,
      exp: now + 3600,
      email,
      email_verified: false,
      firebase: {
        identities: { email: [TEST_UID] },
        sign_in_provider: "password",
      },
    }),
    "unsigned-test-signature",
  ].join(".");
}

/** Accounts this fake backend knows about, seeded with one login. */
const ACCOUNTS = new Map<string, { email: string; password: string }>([
  [TEST_EMAIL, { email: TEST_EMAIL, password: TEST_PASSWORD }],
]);

export interface AuthStubLog {
  resetEmails: string[];
}

export function authStubLog(): AuthStubLog {
  return { resetEmails: [] };
}

const ERRORS: Record<string, unknown> = {
  invalidCredential: { error: { code: 400, message: "INVALID_LOGIN_CREDENTIALS" } },
  emailExists: { error: { code: 400, message: "EMAIL_EXISTS" } },
  weakPassword: {
    error: { code: 400, message: "WEAK_PASSWORD : Password should be at least 6 characters" },
  },
};

/**
 * CORS headers for the stubbed responses.
 *
 * The Firebase SDK calls identitytoolkit from the page, so the browser sends a
 * preflight OPTIONS first. Google answers it with these headers; without them
 * the request fails before the SDK ever sees a body, and every sign-in is
 * reported as "Unable to connect".
 */
const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Max-Age": "3600",
};

function respondJson(req: HTTPRequest, body: unknown, status = 200): void {
  void req.respond({
    status,
    contentType: "application/json",
    headers: CORS,
    body: JSON.stringify(body),
  });
}

/** Routes one intercepted request. Returns true when it was handled. */
export function routeAuthRequest(req: HTTPRequest, log: AuthStubLog): boolean {
  const url = req.url();
  const path = url.split("?")[0];

  const identity = path.includes("/v1/accounts:signInWithPassword");
  const signUp = path.includes("/v1/accounts:signUp");
  const lookup = path.includes("/v1/accounts:lookup");
  const oob = path.includes("/v1/accounts:sendOobCode");
  const token = url.includes("securetoken.googleapis.com");

  const isAuthCall = identity || signUp || lookup || oob || token;
  if (!isAuthCall) return false;

  // Preflight. Answered before anything else so the browser lets the real
  // request through.
  if (req.method() === "OPTIONS") {
    void req.respond({ status: 204, headers: CORS, body: "" });
    return true;
  }

  if (token) {
    // Token refresh. Returning the same still-valid token stops the SDK
    // retrying in a loop during the run.
    respondJson(req, {
      access_token: makeIdToken(TEST_EMAIL),
      token_type: "Bearer",
      expires_in: "3600",
      refresh_token: "test-refresh-token",
      user_id: TEST_UID,
    });
    return true;
  }

  let payload: Record<string, string> = {};
  try {
    payload = JSON.parse(req.postData() ?? "{}");
  } catch {
    /* empty body is fine for lookup */
  }

  if (identity || signUp) {
    const email = String(payload.email ?? "").toLowerCase();
    const password = String(payload.password ?? "");

    if (identity) {
      const acct = ACCOUNTS.get(email);
      if (!acct || acct.password !== password) {
        respondJson(req, ERRORS.invalidCredential, 400);
        return true;
      }
    } else {
      if (ACCOUNTS.has(email)) {
        respondJson(req, ERRORS.emailExists, 400);
        return true;
      }
      if (password.length < 6) {
        respondJson(req, ERRORS.weakPassword, 400);
        return true;
      }
      ACCOUNTS.set(email, { email, password });
    }

    respondJson(req, {
      kind: "identitytoolkit#SignInWithPasswordResponse",
      localId: TEST_UID,
      email,
      idToken: makeIdToken(email),
      refreshToken: "test-refresh-token",
      expiresIn: "3600",
      emailVerified: false,
    });
    return true;
  }
  if (lookup) {
    respondJson(req, {
      kind: "identitytoolkit#GetAccountInfoResponse",
      users: [
        {
          localId: TEST_UID,
          email: TEST_EMAIL,
          emailVerified: false,
          validSince: String(Math.floor(Date.now() / 1000) - 3600),
          lastLoginAt: String(Math.floor(Date.now() / 1000)),
          createdAt: String(Math.floor(Date.now() / 1000) - 86400),
          passwordHash: "",
        },
      ],
    });
    return true;
  }

  if (oob) {
    const email = String(payload.request ?? payload.email ?? "");
    log.resetEmails.push(email);
    respondJson(req, {
      kind: "identitytoolkit#GetOobConfirmationCodeResponse",
      email,
    });
    return true;
  }

  return false;
}

/**
 * Intercepts Firebase traffic on `page` and passes everything else through.
 * Must be installed BEFORE the first navigation.
 *
 * `debug` logs every Firebase URL so a failing sign-in can be diagnosed
 * without guessing; it is off unless explicitly requested.
 */
export async function installAuthBackend(
  page: Page,
  log: AuthStubLog,
  debug = false
): Promise<void> {
  await page.setRequestInterception(true);
  page.on("request", (req) => {
    const handled = routeAuthRequest(req, log);
    if (debug && /googleapis\.com/.test(req.url())) {
      console.log(`    [auth-stub] ${handled ? "ROUTE" : "pass "} ${req.url().slice(0, 110)}`);
    }
    if (handled) return;
    void req.continue();
  });
}

/** Fills the login form and submits, driving the real LoginScreen. */
export async function signInViaUi(
  page: Page,
  email = TEST_EMAIL,
  password = TEST_PASSWORD
): Promise<void> {
  await page.waitForSelector('input[type="email"]', { timeout: 30_000 });
  await page.type('input[type="email"]', email);
  await page.type('input[type="password"]', password);
  await page.click('form button[type="submit"]');
}

/**
 * Clicks a link/button whose text matches `re`.
 *
 * The click is dispatched inside the page rather than through an ElementHandle:
 * an ElementHandle is bound to the page's main frame via `this`, which loses
 * its type context and trips TS2684. A plain DOM click avoids the handle
 * entirely and is sufficient for a real user click simulation.
 */
export async function clickByText(page: Page, re: RegExp): Promise<boolean> {
  return page.evaluate((src) => {
    const rx = new RegExp(src, "i");
    const node = Array.from(document.querySelectorAll("button, a")).find((n) =>
      rx.test((n.textContent || "").trim())
    ) as HTMLElement | undefined;
    if (!node) return false;
    node.click();
    return true;
  }, re.source);
}

/** Switches the LoginScreen to a given mode via its footer links. */
export async function switchMode(page: Page, mode: "signup" | "forgot"): Promise<boolean> {
  if (mode === "signup") return clickByText(page, /create an account/i);
  return clickByText(page, /forgot password/i);
}

