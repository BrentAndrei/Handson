# HandsOn — Filipino Sign Language Live Recognition

## File structure

```
handson/
  index.html
  package.json
  vite.config.ts
  tsconfig.json
  tsconfig.node.json
  public/
    sw.js                 # Service worker: caches model + WASM, network-first otherwise
    manifest.json         # PWA manifest (dark theme)
    favicon.svg           # App icon (cyan "H")
    models/fsl_model/     # TF.js model (float32, ~2 MB)
      model.json
      group1-shard1of1.bin
  src/
    main.tsx              # React entry point
    App.tsx               # Main layout, adaptive resolution ladder, hero header
    index.css             # Design system: dark navy + cyan, glass panels, skeletons
    vite-env.d.ts
    hooks/
      useCamera.ts        # getUserMedia lifecycle (start/stop/resolution)
      useHolisticPipeline.ts # MediaPipe HolisticLandmarker + 24fps rAF capture loop (main thread)
    utils/
      normalizeLandmarks.ts # Translation + isotropic scale normalization
    mediapipe/
      pack.ts             # Packs raw landmark results into feature vectors
      drawResult.ts       # Draws pose/hand/face skeleton overlay on canvas
      smoothing.ts        # Temporal smoothing + extrapolation for display stability
      types.ts            # TypeScript types for landmark results
    inference/
      useSignInference.ts # TF.js model: 345-dim features → sign label + confidence
    features/
      datasetRecorder.ts  # Collects labeled clip frames for training data
      featureSelect.ts    # Selects 345-dim feature vector from normalized landmarks
      grammarParser.ts    # Rule-based FSL gloss → English sentence (disambiguation + reorder)
      sentenceBuilder.ts  # Accumulates gloss tokens into sentences with End/Clear
      windowBuffer.ts     # Ring buffer for sliding 16-frame feature windows
    components/
      CameraPreview.tsx    # Video + canvas overlay, glass frame, skeleton loading
      StatusBar.tsx        # System status pills (camera, model, tier, fps) with tooltips
      PredictionOverlay.tsx # Live prediction hero + confidence bar + skeleton
      SentenceOverlay.tsx  # Sentence mode, English/FSL toggle, recent history
      RecordingControls.tsx # Dataset collection (dev-only, gated on ?dev=1)
      AccessibilityControls.tsx # Text size + high contrast toggles (inline bottom-right)
      Skeleton.tsx         # Shimmer loading placeholders
      Tooltip.tsx          # Hover/focus info icon tooltips
```

## Run it

```bash
npm install
npm run dev
```

Then open the printed `localhost` URL (default `https://localhost:5173` — HTTPS required for camera on LAN).
Grant camera permission. The status bar under the video shows pipeline readiness and measured fps.
The first load pauses a few seconds while the MediaPipe model downloads from Google's CDN.

Requires Node 18+ and a browser with `OffscreenCanvas`/`createImageBitmap` support (all current Chrome, Edge, Firefox, Safari 17+).

## Architecture

**Camera → Recognition pipeline:**

1. `CameraPreview` requests `getUserMedia` at the active resolution tier.
2. `useHolisticPipeline` runs MediaPipe `HolisticLandmarker` on the main thread at 24fps (rAF + accumulator pacing), producing pose/hand/face landmarks.
3. Landmarks flow through `normalizeLandmarks` (scale/translation) → `featureSelect` (345-dim vector) → `useSignInference` (TF.js model) → predicted gloss label.
4. `SentenceOverlay` accumulates gloss tokens; `grammarParser.ts` converts them to English via canonicalization, disambiguation, and Topic-Comment reordering.
5. Adaptive resolution ladder: 960×540 → 640×480 → 424×240, auto-downgraded when fps stays below 12 for 3 seconds (sustained only, no auto-up).

**Performance (main thread, no worker):**

- rAF + accumulator pacing, not `setInterval` — self-corrects for jitter, pauses in background tabs.
- Explicit backpressure: a slow frame drops a frame, not a growing queue.
- Display smoothing and extrapolation are isolated from the feature pipeline so stale/holding points never leak into model input.
- No web worker — MediaPipe `HolisticLandmarker` runs on the main thread with `createImageBitmap` decode.

## PWA

The app is installable: `manifest.json` (dark theme), `favicon.svg` icon, and `sw.js` (caches the model + WASM for offline use). Everything else is network-first so new builds ship without cache-busting.

## Known extension points

- `windowBuffer.ts` currently holds 16-frame sliding windows for inference. If you later need every frame for offline recording, add a separate mode that queues to IndexedDB.
- The inference downscale layer (`?downscale=0` query param) is UNVALIDATED — no real fps measurement has been performed. It reduces the model input resolution below the camera resolution for potential speed gains on slow devices.
- Grammar parser is rule-based (canonicalization + disambiguation + reordering + phrase templates). It does not invent words or insert grammar not present in the input gloss sequence.
