# HandsOn — Filipino Sign Language Live Recognition

A browser-based, bidirectional **Sign-to-Text** and **Text-to-Sign** interface for
Filipino Sign Language (FSL). Everything runs client-side: no video leaves the
machine, and there is no backend to deploy.

## Requirements

- **Node.js 18+** (developed on 24.19.0) and npm (11.17.0)
- Chrome or Edge, latest version, with WebGL
- A webcam

## Run it

```bash
npm install
npm run dev
```

Vite prints a local and a LAN URL. Open the **https** one and accept the
self-signed certificate warning — HTTPS is required because browsers withhold
`getUserMedia()` on plain `http://` for anything that is not `localhost`.

Grant camera permission when prompted. The status bar shows camera, model and
resolution-tier readiness. The first load pauses briefly while the MediaPipe
model fetches from Google's CDN.

To produce a production bundle:

```bash
npm run build      # -> dist/
npm run preview
```

## Verify it

```bash
npm run verify:all
```

Runs the full gate suite: typechecks, the contrast palette contract, the
retargeting suite (37 tests / 2,789 checks), root-translation checks, a
**bit-identical golden-output comparison** (7,940 floats), a production build,
and the headless UI smoke check.

Individual gates, useful when iterating:

| Command | What it proves |
| --- | --- |
| `npm run verify:retarget` | Landmark to bone retargeting correctness |
| `npm run verify:golden` | Output is bit-identical to the recorded baseline |
| `npm run verify:root` | Root translation / grounding |
| `npm run verify:footing` | No ground penetration, feet and ankles untouched |
| `npm run verify:live` | Live to recorded switch is clean, zero hot-path allocation |
| `npm run verify:zeroalloc` | No per-frame THREE geometry allocation |
| `npm run verify:handedness` | MediaPipe handedness is not silently mirrored |
| `npm run verify:contrast` | Every text/background pair meets WCAG AA |
| `npm run verify:ui` | Renders the real build headless and screenshots it |

The UI check needs a Chromium-based browser; it looks for Edge or Chrome in
their standard install locations.

## Layout

```
src/
  App.tsx              App shell: home / translate / avatar / account views,
                       adaptive resolution ladder, hero status bar
  index.css            Design system: neo-brutalist tokens, paper grid,
                       hard shadows, display type scale
  screens/             Home, Login, Account
  components/
    ui/                BrutalNav, BrutalCard, BrutalButton, BrutalBadge,
                       BrutalLabel, BrutalSticker, BrutalArrow, BrutalInput
    mascots/           EyeMascot, HandsMascot (pointer-reactive)
    CameraPreview, StatusBar, PredictionOverlay, SentenceOverlay,
    AccessibilityControls, Tooltip, SignAvatar
  hooks/               useCamera, useHolisticPipeline
  inference/           useSignInference — TF.js GRU, 345-dim input -> gloss
  features/            featureSelect, windowBuffer, grammarParser,
                       sentenceBuilder, datasetRecorder
  avatar/              retargeter, handRetargeter, boneResolver, canonicalPose,
                       coordinateSystem, constraints, footingSolver,
                       rootTranslation, mediapipeAdapter, smplxAdapter
  mediapipe/           pack, drawResult, smoothing, types
  lib/                 cn, usePointerIntent
public/
  models/fsl_model/    TF.js bidirectional GRU (model.json + shard)
  models/              Xbot / RobotExpressive / Soldier GLB avatars
  avatar-data/         Pre-recorded sign sequences, one JSON per gloss
  avatar-data-smplx/   SMPL-X retargeted data (regenerated, not committed)
scripts/               Verification gates and diagnostic tooling
signavatars-service/   Optional Python SMPL-X fitting service
training/              Dataset extraction and model training
```

## Architecture

**Sign-to-Text.** `CameraPreview` requests `getUserMedia` at the active
resolution tier. `useHolisticPipeline` runs MediaPipe `HolisticLandmarker` on
the main thread at 24fps (rAF with accumulator pacing), producing 543
pose/hand/face landmarks. These are reduced to a **345-value** vector per
frame — pose, both hands, and a 40-point subset of the lips — and pushed
through a **16-frame sliding window** into a **bidirectional GRU** (TF.js),
which classifies the window as one isolated sign. A heuristic layer detects
pauses between signs and assembles recognised words into sentences shown as
live captions.

**Text-to-Sign.** The user types an English sentence. A rule-based parser
reorders it into FSL gloss — question fronting, negation, and selected
pronoun/copula patterns — then drives the 3D avatar through the pre-recorded
sequences in `public/avatar-data/`.

**Adaptive resolution.** 960x540 -> 640x480 -> 424x240, auto-downgraded when
the measured frame rate stays below threshold for a sustained period. It never
auto-upgrades, to avoid oscillation. The active tier is shown in the status
bar.

**Performance (main thread, no worker).** rAF + accumulator pacing rather than
`setInterval`, so it self-corrects for jitter and pauses cleanly in background
tabs. Explicit backpressure: a slow frame drops a frame instead of growing a
queue. Display smoothing and extrapolation are isolated from the feature
pipeline, so stale or held points never leak into model input.

## Accessibility

Text size (A / AA / AAA) and high-contrast toggles are built in and persist.
Every interactive element has a visible focus ring, and status is never
communicated by colour alone — badges carry text labels, and the active nav
item sets `aria-current`.

## Known limitations

- Recognises **isolated** signs, not continuous co-articulated signing; the
  sign-to-sentence assembly relies on pauses between signs.
- Filipino Sign Language only.
- No non-manual markers: the parser reorders and punctuates but does not add
  eyebrow or facial-expression information.
- Two-handed and finger-dense signs are harder to classify.
- Vocabulary is limited to the trained dataset.
- Recognition runs on the main thread, so lower-end hardware sees lower frame
  rates.

## Related docs

- `ARCHITECTURE.md` — deeper design notes
- `AGENTS.md` — conventions and optional service setup

