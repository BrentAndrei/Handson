import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

/**
 * Registers the service worker if the browser supports it, and exposes a
 * global `offline` flag that App.tsx reads to show the offline banner.
 *
 * The SW is best-effort: if registration fails (no HTTPS in dev, restricted
 * context), the app still works normally — it just won't cache the heavy
 * assets for offline use.
 */
function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker
    .register("/sw.js")
    .then((reg) => {
      reg.addEventListener("updatefound", () => {
        const worker = reg.installing;
        if (!worker) return;
        worker.addEventListener("statechange", () => {
          if (worker.state === "installed" && navigator.onLine) {
            (window as any).__sw_ready__ = true;
          }
        });
      });
    })
    .catch(() => {
      /* best-effort: ignore */
    });
}

(window as any).__offline__ = typeof navigator !== "undefined" ? !navigator.onLine : false;
window.addEventListener("online", () => {
  (window as any).__offline__ = false;
  window.dispatchEvent(new Event("offline-change"));
});
window.addEventListener("offline", () => {
  (window as any).__offline__ = true;
  window.dispatchEvent(new Event("offline-change"));
});

registerServiceWorker();
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
