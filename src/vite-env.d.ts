/// <reference types="vite/client" />

/**
 * Firebase Web config, supplied by the .env file (see .env.example).
 * Declared here so `import.meta.env.VITE_FIREBASE_*` is type-safe.
 */
interface ImportMetaEnv {
  readonly VITE_FIREBASE_API_KEY: string;
  readonly VITE_FIREBASE_AUTH_DOMAIN: string;
  readonly VITE_FIREBASE_PROJECT_ID: string;
  readonly VITE_FIREBASE_STORAGE_BUCKET: string;
  readonly VITE_FIREBASE_MESSAGING_SENDER_ID: string;
  readonly VITE_FIREBASE_APP_ID: string;
  readonly VITE_DEBUG_INFERENCE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

