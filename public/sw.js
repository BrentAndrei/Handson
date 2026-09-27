/**
 * HandsOn service worker.
 *
 * Caches the two large static assets that dominate load time and would
 * otherwise refetch on every visit: the MediaPipe WASM bundle and the TF.js
 * model. Everything else is network-first so the app always sees the latest
 * code after a deploy.
 *
 * Install: precache the heavy assets.
 * Fetch: cache-first for the two known assets, network-first for everything
 *         else (so a new build ships without a manual cache-bust).
 */

const CACHE_NAME = "handson-v3";
const PRECACHE = [
  "/models/fsl_model/model.json",
  "/models/fsl_model/group1-shard1of1.bin",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      // Failures here shouldn't kill the SW — the app still works online.
      await Promise.allSettled(PRECACHE.map((url) => fetch(url).then((r) => cache.put(url, r.clone()))));
      await self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)));
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const isPrecachedAsset = PRECACHE.some((p) => url.pathname === p);

  if (isPrecachedAsset) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(req);
        if (cached) return cached;
        const fresh = await fetch(req);
        cache.put(req, fresh.clone());
        return fresh;
      })()
    );
    return;
  }

  event.respondWith(
    fetch(req).catch(async () => {
      const cache = await caches.open(CACHE_NAME);
      return cache.match(req);
    })()
  );
});