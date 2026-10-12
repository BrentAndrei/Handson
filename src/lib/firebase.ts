/**
 * FIREBASE CONFIGURATION
 * ---------------------------------------------------------------------------
 * Single place where the Firebase Web SDK is initialised. Every value comes
 * from a Vite environment variable (VITE_FIREBASE_*) so nothing is hardcoded
 * and the real .env stays out of version control.
 *
 * SECURITY NOTE
 * -------------
 * These values (apiKey, authDomain, projectId, ...) are *web app* config, not
 * secrets. The Firebase Web API key is designed to ship inside a browser
 * bundle; access is controlled by Authentication rules and authorised domains
 * in the Firebase console, not by hiding the key. Service-account private keys
 * and the Firebase Admin SDK must NEVER appear here or anywhere in this
 * frontend — that material grants unrestricted server-side access.
 */
import { getApp, getApps, initializeApp, type FirebaseApp } from "firebase/app";
import {
  browserLocalPersistence,
  getAuth,
  setPersistence,
  type Auth,
} from "firebase/auth";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY ?? "",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN ?? "",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID ?? "",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ?? "",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID ?? "",
  appId: import.meta.env.VITE_FIREBASE_APP_ID ?? "",
};

/**
 * True only when every required key is present.
 *
 * Checked rather than assumed: Vite replaces a missing `VITE_*` variable with
 * `undefined` at build time, so an app built without a .env would otherwise
 * initialise Firebase with blanks and fail deep inside the SDK with an opaque
 * message. The auth gate turns this into a readable setup screen instead.
 */
export function isFirebaseConfigured(): boolean {
  return Boolean(
    firebaseConfig.apiKey &&
      firebaseConfig.authDomain &&
      firebaseConfig.projectId &&
      firebaseConfig.appId
  );
}

/** Lazily creates the app so importing this module has no side effects. */
export function getFirebaseApp(): FirebaseApp {
  return getApps().length ? getApp() : initializeApp(firebaseConfig);
}

let authPromise: Promise<Auth> | null = null;

/**
 * Returns the Firebase Auth instance with browser-local persistence explicitly
 * requested.
 *
 * Persistence is what makes "log in, refresh the browser, still logged in"
 * work: the SDK keeps its own signed-in credential in IndexedDB and rehydrates
 * the session on the next load. HandSon stores nothing itself.
 *
 * `getAuth` validates the API key and THROWS SYNCHRONOUSLY when it is missing
 * or malformed, so the throw has to be caught here rather than deferred into
 * the promise — otherwise it escapes as a render-time exception and blanks the
 * app. It is converted into a rejected promise so every caller sees one
 * uniform failure path.
 */
export function getFirebaseAuth(): Promise<Auth> {
  if (!authPromise) {
    try {
      const auth = getAuth(getFirebaseApp());
      authPromise = setPersistence(auth, browserLocalPersistence)
        .catch(() => auth) // e.g. storage blocked in private mode — session just won't survive a reload
        .then(() => auth);
    } catch (err) {
      authPromise = Promise.reject(
        err instanceof Error ? err : new Error(String(err))
      );
      // Allow a later attempt to retry instead of caching the rejection.
      authPromise.catch(() => {
        authPromise = null;
      });
    }
  }
  return authPromise;
}
