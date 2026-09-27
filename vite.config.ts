import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [
    react(),
    // Only needed so a phone on the same Wi-Fi can open this over HTTPS —
    // browsers treat plain http:// on a LAN IP as an insecure context and
    // silently withhold getUserMedia there (navigator.mediaDevices is
    // undefined), even though the exact same code works fine on
    // http://localhost. Auto-generates a self-signed cert on first run;
    // your phone's browser will show a "not private" warning the first
    // time you open the LAN URL — tap through/"advance" past it, that's
    // expected for a self-signed dev cert and only affects this dev server.
    basicSsl(),
  ],
  server: {
    // MediaPipe's WASM assets are fetched from a CDN by default (see
    // FilesetResolver.forVisionTasks in useHolisticPipeline.ts), so no
    // special COOP/COEP headers are required for this setup. If you later
    // switch to self-hosting the wasm/binarypb assets, keep them under
    // /public so Vite serves them as static files without needing a CDN
    // at all.
    port: 5173,
    // Bind to all interfaces (not just localhost) so the dev server is
    // reachable from other devices on the same network, e.g. a phone
    // testing camera access. `vite` will print both the local and LAN
    // URLs on startup — use the "Network:" one on your phone.
    host: true,
  },
});
