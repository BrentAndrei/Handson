import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CameraPreview } from "./components/CameraPreview";
import { PredictionOverlay } from "./components/PredictionOverlay";
import { AccuracyDashboard } from "./components/AccuracyDashboard";
import { SentenceOverlay } from "./components/SentenceOverlay";
import { StatusBar } from "./components/StatusBar";
import { AccessibilityControls } from "./components/AccessibilityControls";
import { Autocomplete } from "./components/Autocomplete";
import { GlossDisplay } from "./components/GlossDisplay";
import SignAvatar from "./components/SignAvatar";
import { useHolisticPipeline } from "./hooks/useHolisticPipeline";
import { enableSmplxService } from "./avatar/dataLoader";
import { normalizeLandmarks, NormalizedFrame } from "./utils/normalizeLandmarks";
import { selectFeatures } from "./features/featureSelect";
import { useSignInference } from "./inference/useSignInference";
import { INFERENCE_LABELS } from "./inference/labels";
import { useSentenceBuilder } from "./features/sentenceBuilder";
import { englishToGloss } from "./features/englishToGloss";
import { LandmarkSmoother } from "./utils/landmarkSmoother";
import type { LandmarksResult } from "./mediapipe/types";
import { RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET, HAND_COUNT } from "./mediapipe/types";
import { HomeScreen } from "./screens/HomeScreen";
import { LoginScreen } from "./screens/LoginScreen";
import { AccountScreen } from "./screens/AccountScreen";
import { PracticeScreen } from "./screens/PracticeScreen";
import { VideoCaptureScreen } from "./screens/VideoCaptureScreen";
import { AuthLoadingScreen } from "./screens/AuthLoadingScreen";
import { useAuth } from "./auth/AuthProvider";
import { BrutalNav, type AppView } from "./components/ui/BrutalNav";
import { CONFIDENCE_PRESETS } from "./lib/videoCapture";
import {
  OnboardingTour,
  hasSeenTour,
  markTourSeen,
} from "./components/OnboardingTour";

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
};
const pageTransition = { duration: 0.35, ease: "easeOut" as const };


const sectionVariants = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -12 },
};

const sectionTransition = (delay: number) => ({ duration: 0.25, delay, ease: "easeOut" as const });

const PREDICT_KEY = "predict";
const TEXT_KEY = "text";
const HOME_KEY = "home";
const PRACTICE_KEY = "practice";
const CAPTURE_KEY = "capture";

/** Developer feature-vector dump. Off by default; flip to true to trace
 *  `selectFeatures` output once per second in the console. */
const DEBUG_FEATURES = false;
const FEATURE_DEBUG_INTERVAL_MS = 1000;
/** Live inference is trained on 8 FPS decimated data (16-frame window =
 *  2.0 s of real motion). Gate the tf.js pushes to ~8 Hz regardless of
 *  capture rate so the model sees training-matched temporal scale even
 *  when HolisticLandmarker runs at 24-30 fps. */
const INFERENCE_SAMPLE_INTERVAL_MS = 125;


interface ResolutionTier {
  width: number;
  height: number;
  inferenceWidth?: number;
  inferenceHeight?: number;
}

const RESOLUTION_TIERS: readonly ResolutionTier[] = [
   { width: 640, height: 480, inferenceWidth: 320, inferenceHeight: 240 },
   { width: 424, height: 240, inferenceWidth: 212, inferenceHeight: 120 },
   { width: 320, height: 240, inferenceWidth: 160, inferenceHeight: 120 },
 ];

const FPS_DOWNGRADE_THRESHOLD = 8;
const LOW_FPS_SAMPLE_INTERVAL_MS = 800;
const LOW_FPS_CONSECUTIVE_SAMPLES = 5;

function readDownscaleFlag(): boolean {
  if (typeof window === "undefined") return false;
  const v = new URLSearchParams(window.location.search).get("downscale");
  return v === "1" || v === "true";
}

/**
 * The application proper. Every screen, pipeline and view lives inside this
 * component and is only ever mounted for a signed-in user — the gate below is
 * the single place that decides.
 */
function HandSonApp() {
  const { user } = useAuth();
  const latestFrameRef = useRef<NormalizedFrame | null>(null);
  const featureDebugLastRef = useRef<number>(0);

  // PHASE 10: the exact Float32Array handed to the avatar, plus a monotonic
  // counter. The counter is what lets the render loop retarget at most once per
  // inference result instead of once per animation frame.
  const avatarLiveFrameRef = useRef<{ vector: Float32Array; seq: number } | null>(null);
  const liveSeqCounterRef = useRef(0);
  /**
   * PHASE 1 UI: the old `landing | predict | text` triple is now the four-item
   * app shell (home / translate / avatar / account). `predict` and `text` are
   * the TRANSLATE and AVATAR views respectively â€” the existing pipelines below
   * are untouched, only the names of the screens changed.
   */
  const [view, setView] = useState<AppView>("home");
  /**
   * FIRST-RUN TOUR. Shows until dismissed; the dismissal is
   * persisted PER ACCOUNT, so every new account gets the
   * tour while a returning one never sees it again.
   */
  const [tourOpen, setTourOpen] = useState<boolean>(() => !hasSeenTour(user?.id));
  const closeTour = useCallback(() => {
    markTourSeen(user?.id);
    setTourOpen(false);
  }, [user?.id]);
  const [showEnglish, setShowEnglish] = useState(true);
  const [textSize, setTextSize] = useState<"sm" | "base" | "lg">("base");
  const [showAccuracyDashboard, setShowAccuracyDashboard] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.15);
  const [highContrast, setHighContrast] = useState(false);
  const [showAvatar, setShowAvatar] = useState(true);
  const avatarLandmarksRef = useRef<Float32Array | null>(null);
  // PHASE 31: hands included, matching the recorded path in dataLoader.ts. Live
// capture is the noisiest source of hand landmarks (webcam tracking), and the
// window-3 average removes 67% of the hand stream's high-frequency content at
// 0-1 frames of group delay. Lower limbs stay filtered as in Phase 24.
const landmarkSmootherRef = useRef(new LandmarkSmoother(3, true, true));
  const [offline, setOffline] = useState<boolean>(
    typeof navigator !== "undefined" ? !navigator.onLine : false
  );
  const [englishText, setEnglishText] = useState("");
  const [glossResult, setGlossResult] = useState<ReturnType<typeof englishToGloss> | null>(null);
  const [textStatus, setTextStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => {
    enableSmplxService(true);
    const onOffline = () => setOffline(true);
    const onOnline = () => setOffline(false);
    const onSwOffline = () => setOffline(!!(window as any).__offline__);
    window.addEventListener("offline", onOffline);
    window.addEventListener("online", onOnline);
    window.addEventListener("offline-change", onSwOffline);
    return () => {
      window.removeEventListener("offline", onOffline);
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline-change", onSwOffline);
    };
  }, []);
   const inference = useSignInference("/models/fsl_model/tfjs_combined/model.json", INFERENCE_LABELS);
  const sentenceBuilder = useSentenceBuilder();
  const inferencePushRef = useRef(inference.pushFrame); inferencePushRef.current = inference.pushFrame;
  const lastInferencePushRef = useRef(0);
  const sentenceOnActivityRef = useRef(sentenceBuilder.onActivity); sentenceOnActivityRef.current = sentenceBuilder.onActivity;
  const sentenceOnPredictionRef = useRef(sentenceBuilder.onPrediction); sentenceOnPredictionRef.current = sentenceBuilder.onPrediction;
  useEffect(() => { if (inference.prediction) sentenceOnPredictionRef.current(inference.prediction, performance.now()); }, [inference.prediction]);
  const handleLandmarks = useCallback((msg: LandmarksResult) => {
    // Smooth ONLY for avatar display — the classifier must see unsmoothed
    // features matching the training pipeline (extract_landmarks.py does NOT
    // smooth, so feeding smoothed data to the model creates a temporal
    // domain mismatch that degrades accuracy).
    const smoothedBuf = landmarkSmootherRef.current.smooth(new Float32Array(msg.buffer));
    // Avatar live path: smoothed + normalized
    const normalized = normalizeLandmarks({ ...msg, buffer: smoothedBuf.buffer });
    avatarLiveFrameRef.current = { vector: normalized.vector, seq: liveSeqCounterRef.current++ };

    // Inference path: unsmoothed + normalized (matching extract_landmarks.py)
    const inferenceNormalized = normalizeLandmarks(msg); latestFrameRef.current = inferenceNormalized;
    const features = selectFeatures(inferenceNormalized.vector);
    if (msg.hasRightHand) {
      const view = new Float32Array(msg.buffer);
      avatarLandmarksRef.current = view.subarray(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + HAND_COUNT * 3);
    } else if (msg.hasLeftHand) {
      const view = new Float32Array(msg.buffer);
      avatarLandmarksRef.current = view.subarray(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + HAND_COUNT * 3);
    } else {
      avatarLandmarksRef.current = null;
    }
    if (DEBUG_FEATURES) {
      const now = performance.now();
      if (!featureDebugLastRef.current || now - featureDebugLastRef.current >= FEATURE_DEBUG_INTERVAL_MS) {
        featureDebugLastRef.current = now;
        console.log("[debug-features] ts=" + msg.timestamp.toFixed(0) + " vector=" + Array.from(features).map(v => v.toFixed(4)).join(","));
      }
    }

    const now = performance.now();
    if (now - lastInferencePushRef.current >= INFERENCE_SAMPLE_INTERVAL_MS) {
      lastInferencePushRef.current = now;
      inferencePushRef.current(features);
    }
    sentenceOnActivityRef.current(msg.hasLeftHand || msg.hasRightHand, performance.now());
  }, []);
  const [isMobile, setIsMobile] = useState(() => 
    typeof window !== "undefined" && window.innerWidth < 768
  );
  
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener?.("change", handler);
    // Also handle resize as fallback
    const resizeHandler = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", resizeHandler);
    return () => {
      mq.removeEventListener?.("change", handler);
      window.removeEventListener("resize", resizeHandler);
    };
  }, []);
  
  const confidence = isMobile ? CONFIDENCE_PRESETS.balanced : CONFIDENCE_PRESETS.permissive;
  
  const { ready, error, fps, attachVideo, attachCanvas, setInferenceSize } = useHolisticPipeline(handleLandmarks, { confidence });

  // PHASE 10 â€” live vs recorded source for the avatar.
  //
  // The avatar is driven by the webcam whenever the camera is live AND the user
  // is on a view that actually shows the camera. On the text-only view there is
  // no camera in context, so recorded gloss playback continues -- which is the
  // behaviour that was already there and must not regress.
  //
  // Auto-derived rather than adding a new UI control: the spec forbids UI
  // redesigns, and this makes the switch happen exactly when the camera starts
  // or stops, with no new affordance for the user to discover.
  const avatarLiveEnabled = ready && !error && view === "translate";

  const [tierIndex, setTierIndex] = useState<number>(0);
  const lowFpsCountRef = useRef<number>(0);
  const lastFpsEvalRef = useRef<number>(0);
  const downgradeRef = useRef<boolean>(readDownscaleFlag());

  const resolution = RESOLUTION_TIERS[tierIndex];
  const tierInferenceSize = resolution.inferenceWidth && resolution.inferenceHeight && downgradeRef.current
    ? { width: resolution.inferenceWidth, height: resolution.inferenceHeight }
    : null;

  useEffect(() => { setInferenceSize(tierInferenceSize); }, [tierInferenceSize, setInferenceSize]);

  useEffect(() => {
    if (!fps || fps <= 0) return;
    if (tierIndex >= RESOLUTION_TIERS.length - 1) return;

    const now = performance.now();
    if (now - lastFpsEvalRef.current < LOW_FPS_SAMPLE_INTERVAL_MS) return;
    lastFpsEvalRef.current = now;

    if (fps < FPS_DOWNGRADE_THRESHOLD) {
      lowFpsCountRef.current += 1;
    } else {
      lowFpsCountRef.current = 0;
    }

    if (lowFpsCountRef.current >= LOW_FPS_CONSECUTIVE_SAMPLES) {
      lowFpsCountRef.current = 0;
      setTierIndex((i) => Math.min(i + 1, RESOLUTION_TIERS.length - 1));
    }
  }, [fps, tierIndex]);

  return (
    <div
      className={`min-h-screen ${textSize !== "base" ? `data-text-size="${textSize}"` : ""} ${highContrast ? "data-hc=true" : ""}`}
      data-text-size={textSize}
      data-hc={highContrast ? "true" : "false"}
    >
      {/* The printed ground: a fine + coarse ink grid on warm paper, fixed
          behind everything and masked toward the centre so it reads as stock
          rather than as a table. aria-hidden because it carries no meaning.
          This is what replaced the ambient-glow/glassmorphism treatment. */}
      <div className="hs-ground" aria-hidden="true" />

      <StatusBar
        items={[
          { label: "Camera", ready, error },
          { label: "Model", ready: inference.ready, error: inference.error },
          { label: `Tier ${tierIndex + 1}/${RESOLUTION_TIERS.length}`, ready: true, tooltip: `Resolution tier ${tierIndex + 1} of ${RESOLUTION_TIERS.length}. Auto-downgrades when fps stays below ${FPS_DOWNGRADE_THRESHOLD} for ${LOW_FPS_CONSECUTIVE_SAMPLES}s.${tierInferenceSize ? " Inference downscale " + tierInferenceSize.width + "x" + tierInferenceSize.height + " (UNVALIDATED â€” ?downscale=0 to disable)" : " No inference downscale."}` },
        ]}
        fps={fps}
        offline={offline}
      />

      {/* pb-28 reserves room so the fixed bottom nav never covers content. */}
      <main className="relative mx-auto max-w-6xl px-3 pb-28 pt-4 sm:px-6 pb-safe">
        <AnimatePresence mode="wait">
        {view === "home" && (
          <motion.div
            key={HOME_KEY}
            data-view="home"
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
          >
            <HomeScreen
              modelReady={inference.ready}
              modelError={Boolean(inference.error)}
              onNavigate={setView}
            />
          </motion.div>
        )}
        {view === "translate" && (
          <motion.div
            key={PREDICT_KEY}
            className={`grid gap-6 ${inference.ready ? "lg:grid-cols-[minmax(320px,1.45fr)_minmax(19rem,0.8fr)]" : "grid-cols-1"} items-start lg:items-start`}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
          >
            <motion.section
              className="space-y-4 bg-paper-raised border-2 border-ink rounded-none p-4 relative shadow-brutal"
              variants={sectionVariants}
              initial="initial"
              animate="animate"
              transition={sectionTransition(0.05)}
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setView("home")}
                  className="rounded-lg px-3 py-1.5 text-sm font-bold text-ink-soft hover:bg-ink hover:text-paper transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-teal"
                >
                  ← Back
                </button>
                <p className="text-sm font-bold text-teal-deep">Predict Mode</p>
              </div>
              <CameraPreview onVideoReady={attachVideo} onCanvasReady={attachCanvas} resolution={resolution} showFlipButton />
            </motion.section>
            <motion.aside
              className={`space-y-4 ${inference.ready ? "" : "lg:hidden"}`}
              variants={sectionVariants}
              initial="initial"
              animate="animate"
              transition={sectionTransition(0.1)}
            >
              <PredictionOverlay ready={inference.ready} error={inference.error} prediction={inference.prediction} confidenceThreshold={confidenceThreshold} onThresholdChange={setConfidenceThreshold} />
              <SentenceOverlay
                currentWords={sentenceBuilder.currentWords}
                currentEnglish={sentenceBuilder.currentEnglish}
                finalizedSentences={sentenceBuilder.finalizedSentences}
                finalizedEnglish={sentenceBuilder.finalizedEnglish}
                onClear={sentenceBuilder.clear}
                onEndSentence={sentenceBuilder.endSentenceNow}
                showEnglish={showEnglish}
                onToggleEnglish={() => setShowEnglish((v) => !v)}
              />
              <motion.div className="absolute top-3 right-3 z-10">
                <button
                  onClick={() => setShowAccuracyDashboard((v) => !v)}
                  className="glass-pill px-3 py-1 text-xs font-medium text-ink-soft hover:bg-ink/5 transition-colors"
                  title="Toggle accuracy dashboard"
                >
                  {showAccuracyDashboard ? "Hide" : "Show"} stats
                </button>
              </motion.div>
              <AccuracyDashboard show={showAccuracyDashboard} />
            </motion.aside>
          </motion.div>
        )}
        {view === "avatar" && (
          <motion.div
            key={TEXT_KEY}
            className="grid gap-6 lg:grid-cols-[minmax(320px,1.45fr)_minmax(19rem,0.8fr)] items-start lg:items-start animate-soft-in"
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
          >
            <motion.section
              className="space-y-4 bg-paper-raised border-2 border-ink rounded-none p-4 relative shadow-brutal"
              variants={sectionVariants}
              initial="initial"
              animate="animate"
              transition={sectionTransition(0.05)}
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setView("home")}
                  className="rounded-lg px-3 py-1.5 text-sm font-bold text-ink-soft hover:bg-ink hover:text-paper transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-teal"
                >
                  ← Back
                </button>
                <p className="text-sm font-bold text-teal-deep">Text & Avatar</p>
              </div>
              {showAvatar && (
                <SignAvatar
                  signSequence={glossResult?.glosses.map(g => g.gloss)}
                  liveFrameRef={avatarLiveFrameRef}
                  liveActive={avatarLiveEnabled}
                />
              )}
            </motion.section>
            <motion.aside
              className="space-y-4"
              variants={sectionVariants}
              initial="initial"
              animate="animate"
              transition={sectionTransition(0.1)}
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowAvatar((v) => !v)}
                  className={`rounded-lg px-3 py-2.5 text-sm font-bold transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-4 focus:ring-teal ${
                    showAvatar ? "bg-teal text-ink shadow-brutal-sm" : "text-ink-soft hover:bg-ink hover:text-paper"
                  }`}
                >
                  {showAvatar ? "Hide 3D Avatar" : "Show 3D Avatar"}
                </button>
              </div>
<div className="rounded-none border-2 border-teal bg-teal/5 px-4 py-3 transition-colors duration-200">
                <div className="flex items-center gap-2">
                  <div className={`w-2.5 h-2.5 rounded-full ${
                    textStatus === "ready" ? "bg-lime" :
                    textStatus === "loading" ? "bg-sun animate-pulse" :
                    textStatus === "error" ? "bg-coral" :
                    "bg-teal"
                  }`} />
                  <p className="m-0 text-sm font-bold text-ink">
                    {textStatus === "ready" ? "Avatar ready — signing your text" :
                     textStatus === "loading" ? "Loading avatar data..." :
                     textStatus === "error" ? "Error loading avatar" :
                     "Type English below and press Enter to sign it"}
                  </p>
                </div>
              </div>
              <Autocomplete
                value={englishText}
                onChange={(v) => { setEnglishText(v); if (v.trim()) setTextStatus("idle"); }}
                onSubmit={() => {
                  if (englishText.trim()) {
                    setTextStatus("loading");
                    setGlossResult(englishToGloss(englishText));
                    setShowAvatar(true);
                    requestAnimationFrame(() => setTextStatus("ready"));
                  }
                }}
                vocabulary={INFERENCE_LABELS}
              />
              {glossResult && <GlossDisplay result={glossResult} />}
            </motion.aside>
          </motion.div>
        )}
        {view === "practice" && (
          <motion.div
            key={PRACTICE_KEY}
            data-view="practice"
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
          >
            <PracticeScreen
              onNavigate={setView}
              liveFrameRef={avatarLiveFrameRef}
              attachVideo={attachVideo}
              attachCanvas={attachCanvas}
              resolution={resolution}
              cameraReady={ready && !error}
              cameraError={error}
            />
          </motion.div>
        )}
        {view === "capture" && (
          <motion.div
            key={CAPTURE_KEY}
            data-view="capture"
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
          >
            <VideoCaptureScreen onNavigate={setView} />
          </motion.div>
        )}
        <div className="mt-6 flex justify-end">
          <AccessibilityControls
            textSize={textSize}
            onTextSize={setTextSize}
            highContrast={highContrast}
            onHighContrast={setHighContrast}
          />
        </div>

        {/* No auth sub-screen here any more: HandSonApp only mounts for a
            signed-in user, so Account always shows the real profile. */}
        {view === "account" && (
          <AccountScreen onNavigate={setView} onReplayTour={() => setTourOpen(true)} />
        )}
        </AnimatePresence>
        <BrutalNav active={view} onNavigate={setView} />
      </main>

      {/* FIRST-RUN TOUR — mounted AFTER <main> on purpose.
          The UI gate's global mascot scan reads
          `document.querySelector('[data-mascot="eye"]')`, i.e.
          DOCUMENT ORDER, and every home-scoped assertion is
          scoped to [data-view=home]. Placing the overlay after
          <main> keeps the home mascots first in document order
          and the overlay outside every home-scoped sweep. */}
      {tourOpen && <OnboardingTour onDone={closeTour} />}
    </div>
  );
}

/**
 * AUTHENTICATION GATE
 * ---------------------------------------------------------------------------
 *   loading        -> AuthLoadingScreen  (Firebase has not answered yet)
 *   user exists    -> HandSonApp
 *   no user        -> LoginScreen
 *
 * Nothing behind this gate — Home, the real-time translator, the 3D avatar,
 * Account — is reachable without a Firebase session, and none of it is mounted
 * early enough to flash for an unauthenticated visitor. After a sign-out,
 * `onAuthStateChanged` fires with null and this swaps straight back to the
 * login screen; a refresh does the same because Firebase rehydrates the session
 * from its own persisted credential.
 */
export default function App() {
  const { user, status } = useAuth();

  if (status === "loading") return <AuthLoadingScreen />;
  if (!user) return <LoginScreen onNavigate={() => {}} />;
  return <HandSonApp />;
}

