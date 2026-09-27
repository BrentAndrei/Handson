import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CameraPreview } from "./components/CameraPreview";
import { PredictionOverlay } from "./components/PredictionOverlay";
import { SentenceOverlay } from "./components/SentenceOverlay";
import { StatusBar } from "./components/StatusBar";
import { AccessibilityControls } from "./components/AccessibilityControls";
import { Autocomplete } from "./components/Autocomplete";
import { GlossDisplay } from "./components/GlossDisplay";
import SignAvatar from "./components/SignAvatar";
import { useHolisticPipeline } from "./hooks/useHolisticPipeline";
import { normalizeLandmarks, NormalizedFrame } from "./utils/normalizeLandmarks";
import { selectFeatures } from "./features/featureSelect";
import { useSignInference } from "./inference/useSignInference";
import { useSentenceBuilder } from "./features/sentenceBuilder";
import { englishToGloss } from "./features/englishToGloss";
import { LandmarkSmoother } from "./utils/landmarkSmoother";
import type { LandmarksResult } from "./mediapipe/types";
import { RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET, HAND_COUNT } from "./mediapipe/types";

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
};
const pageTransition = { duration: 0.35, ease: "easeOut" as const };

const landingContainer = {
  initial: { opacity: 1 },
  animate: { transition: { staggerChildren: 0.1, delayChildren: 0.08 } },
  exit: { opacity: 0, transition: { duration: 0.15 } },
};
const landingItem = {
  initial: { opacity: 0, y: 16, scale: 0.98 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, y: -8 },
};

const sectionVariants = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -12 },
};

const sectionTransition = (delay: number) => ({ duration: 0.25, delay, ease: "easeOut" as const });

const LANDING_KEY = "landing";
const PREDICT_KEY = "predict";
const TEXT_KEY = "text";

const INFERENCE_LABELS: string[] = [
  "ADOPT",
  "AGAIN",
  "ALLOW",
  "ALRIGHT, NO PROBLEM",
  "APPOINT",
  "ARCHITECT",
  "ARE",
  "ARE YOU A STUDENT?",
  "ARE YOU DEAF",
  "ARE YOU OKAY",
  "ASSIST",
  "ATTEMPT",
  "AUNTIE",
  "BABY",
  "BATH",
  "BEAUTIFUL",
  "BECAUSE",
  "BREAK",
  "BRING",
  "BROTHER",
  "BUY",
  "CALL",
  "CAN",
  "CATCH",
  "CHANGE",
  "CHEF",
  "CHILD",
  "CHILDREN",
  "CHOOSE",
  "CLEAN",
  "COME",
  "COME TO EAT TOGETHER",
  "CONTINUE",
  "CONTROL",
  "COOK",
  "COOL",
  "COPY",
  "COULD",
  "COUNT",
  "COUSIN",
  "CRY",
  "CUTE",
  "DANCE",
  "DANCER",
  "DAUGHTER",
  "DAY",
  "DEAF",
  "DEVELOP",
  "DID",
  "DISTURB",
  "DO",
  "DOCTOR",
  "DRINK",
  "DRIVE",
  "DRUNK",
  "EAT",
  "EIGHT",
  "EIGHTEEN",
  "ELEVEN",
  "EXCHANGE",
  "EXCUSE",
  "FALL",
  "FAMILY",
  "FAT",
  "FATHER",
  "FEEL",
  "FIFTEEN",
  "FIGHT",
  "FINALLY, I MEET YOU",
  "FINGERSPELL",
  "FINISH",
  "FIVE",
  "FOLLOW",
  "FOSTERED",
  "FOUR",
  "FOURTEEN",
  "FRIDAY",
  "FROM",
  "GET",
  "GO",
  "GODFATHER",
  "GODMOTHER",
  "GOOD AFTERNOON",
  "GOOD DAY",
  "GOOD EVENING",
  "GOOD MIDDAY",
  "GOOD MORNING",
  "GOOD NIGHT",
  "GRADUATE",
  "GRANDCHILD",
  "GRANDMA",
  "GRANDPA",
  "HANDSOME",
  "HAPPEN",
  "HAPPY",
  "HAPPY BIRTHDAY",
  "HAPPY FATHERS DAY",
  "HAPPY MOTHERS DAY",
  "HARD OF HEARING",
  "HE/SHE",
  "HE/SHE IS MY CLASSMATE",
  "HE/SHE IS MY SWEETHEART",
  "HEAR",
  "HEARING",
  "HELLO",
  "HELP",
  "HERE",
  "HI",
  "HIS/HER",
  "HOLD",
  "HOW",
  "HOW ARE YOU",
  "HOW ARE YOU?",
  "HOW MUCH",
  "HOW OLD ARE YOU?",
  "HURRY",
  "HUSBAND",
  "I DONT KNOW",
  "I DONT UNDERSTAND",
  "I KNOW A LITTLE SIGN",
  "I KNOW HE/SHE IS MY FRIEND",
  "I UNDERSTAND",
  "I'M FINE",
  "I'M FINE, THANK YOU",
  "I'M GOOD",
  "I'M NOT VERY WELL",
  "I'M OKAY",
  "I'M PRETTY GOOD",
  "I'M SICK",
  "I'M WORKING",
  "I'VE BEEN FINE, THANK YOU",
  "IM PLEASED TO MEET YOU",
  "IMPROVE",
  "IN",
  "INTRODUCE",
  "INTRODUCTION",
  "INVITE",
  "IS",
  "IS IT ALRIGHT",
  "IT'S ALRIGHT",
  "ITS FINE",
  "JUMP",
  "KEEP",
  "KILL",
  "LAUGH",
  "LET",
  "LINE",
  "LISTEN",
  "LIVE",
  "LONG TIME NO SEE",
  "LOOK",
  "MAKE",
  "MARCH",
  "MATCH",
  "MAYBE",
  "ME",
  "MINE",
  "MONDAY",
  "MOTHER",
  "MOVE",
  "MY NAME IS",
  "NEPHEW",
  "NEW",
  "NG",
  "NICE TO MEET YOU",
  "NIECE",
  "NINE",
  "NINETEEN",
  "NO",
  "NO PROBLEM",
  "NO, I AM NOT A STUDENT",
  "NOT BAD",
  "NOT SURE",
  "NOW",
  "NOWHERE",
  "OFF",
  "OH I SEE",
  "OK",
  "OKAY, I GUESS",
  "OLD",
  "ON",
  "ONE",
  "OO",
  "OPO",
  "OUR",
  "OUT",
  "OVER THERE",
  "PARENTS",
  "PAYMENT",
  "PEOPLE",
  "PERMIT",
  "PHOTOGRAPHER",
  "PLAN",
  "PLAY",
  "PLEASE SIGN SLOWLY",
  "PREGNANT",
  "PREPARE",
  "PROBLEM",
  "PROFESSOR",
  "PUT",
  "RESPONSES",
  "RUN",
  "SAD",
  "SATURDAY",
  "SAY",
  "SEARCH",
  "SEE",
  "SEND",
  "SERVE",
  "SEVEN",
  "SEVENTEEN",
  "SHOULD",
  "SING",
  "SISTER",
  "SIT",
  "SIX",
  "SIXTEEN",
  "SLEEP",
  "SMELL",
  "SMILE",
  "SOMEWHERE",
  "SON",
  "SORRY",
  "STAND",
  "STEAL",
  "STEP BROTHER",
  "STEP SISTER",
  "STOP",
  "STUDENT",
  "STUDY",
  "SUNDAY",
  "SUPPORT",
  "SURE",
  "SWIM",
  "TAKE",
  "TAKE CARE",
  "TALK",
  "TEACH",
  "TEACHER",
  "TEENAGER",
  "TEMPT",
  "TEN",
  "THANK YOU",
  "THEIR",
  "THEM",
  "THINK",
  "THIRTEEN",
  "THREE",
  "THURSDAY",
  "TOUCH",
  "TRADE",
  "TRY",
  "TUESDAY",
  "TWELVE",
  "TWENTY",
  "TWO",
  "UGLY",
  "UNCLE",
  "UNDERSTAND",
  "USE",
  "VISIT",
  "WAIT",
  "WAKE UP",
  "WALK",
  "WANT",
  "WAS",
  "WASH",
  "WATCH",
  "WE",
  "WEDDING",
  "WEDNESDAY",
  "WEEK",
  "WELCOME",
  "WERE",
  "WHAT",
  "WHAT ARE YOU STUDYING?",
  "WHAT WORK DO YOU DO?",
  "WHAT'S NEW",
  "WHAT'S YOUR NAME",
  "WHAT'S YOUR NAME?",
  "WHATEVER",
  "WHATS YOUR NAME?",
  "WHEN",
  "WHERE",
  "WHERE DO YOU LIVE?",
  "WHICH",
  "WHICH UNIVERSITY ARE YOU STUDYING AT?",
  "WHISPER",
  "WHO",
  "WHY",
  "WIFE",
  "WILL",
  "WORK",
  "WOULD",
  "WOW",
  "WRITE",
  "YES",
  "YES, I AM A STUDENT",
  "YOU",
  "YOU SIGN FAST",
  "YOUNG",
  "YOUR",
  "Ñ",
];
const DEBUG_FEATURES = true;
const FEATURE_DEBUG_INTERVAL_MS = 1000;

interface ResolutionTier {
  width: number;
  height: number;
  inferenceWidth?: number;
  inferenceHeight?: number;
}

const RESOLUTION_TIERS: readonly ResolutionTier[] = [
  { width: 960, height: 540, inferenceWidth: 480, inferenceHeight: 270 },
  { width: 640, height: 480, inferenceWidth: 320, inferenceHeight: 240 },
  { width: 424, height: 240 },
];

const FPS_DOWNGRADE_THRESHOLD = 12;
const LOW_FPS_SAMPLE_INTERVAL_MS = 1000;
const LOW_FPS_CONSECUTIVE_SAMPLES = 3;

function readDownscaleFlag(): boolean {
  if (typeof window === "undefined") return true;
  const v = new URLSearchParams(window.location.search).get("downscale");
  if (v === "0" || v === "false") return false;
  return true;
}

export default function App() {
  const latestFrameRef = useRef<NormalizedFrame | null>(null);
  const featureDebugLastRef = useRef<number>(0);
  const [view, setView] = useState<"landing" | "predict" | "text">("landing");
  const [showEnglish, setShowEnglish] = useState(true);
  const [textSize, setTextSize] = useState<"sm" | "base" | "lg">("base");
  const [highContrast, setHighContrast] = useState(false);
  const [showAvatar, setShowAvatar] = useState(true);
  const avatarLandmarksRef = useRef<Float32Array | null>(null);
  const landmarkSmootherRef = useRef(new LandmarkSmoother(3));
  const [offline, setOffline] = useState<boolean>(
    typeof navigator !== "undefined" ? !navigator.onLine : false
  );
  const [englishText, setEnglishText] = useState("");
  const [glossResult, setGlossResult] = useState<ReturnType<typeof englishToGloss> | null>(null);
  const [textStatus, setTextStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  function GlassCard({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
    const cardRef = useRef<HTMLDivElement>(null);
    const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
      if (!cardRef.current) return;
      const rect = cardRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;
      cardRef.current.style.setProperty('--mouse-x', `${x}%`);
      cardRef.current.style.setProperty('--mouse-y', `${y}%`);
    }, []);
    return (
      <div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onClick={onClick}
        className="glass-card-interactive cursor-pointer"
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } }}
      >
        {children}
      </div>
    );
  }
  useEffect(() => {
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
  const inference = useSignInference("/models/fsl_model/model.json", INFERENCE_LABELS);
  const sentenceBuilder = useSentenceBuilder();
  const inferencePushRef = useRef(inference.pushFrame); inferencePushRef.current = inference.pushFrame;
  const sentenceOnActivityRef = useRef(sentenceBuilder.onActivity); sentenceOnActivityRef.current = sentenceBuilder.onActivity;
  const sentenceOnPredictionRef = useRef(sentenceBuilder.onPrediction); sentenceOnPredictionRef.current = sentenceBuilder.onPrediction;
  useEffect(() => { if (inference.prediction) sentenceOnPredictionRef.current(inference.prediction, performance.now()); }, [inference.prediction]);
  const handleLandmarks = useCallback((msg: LandmarksResult) => {
    const smoothedBuf = landmarkSmootherRef.current.smooth(new Float32Array(msg.buffer));
    const smoothedMsg = { ...msg, buffer: smoothedBuf.buffer as ArrayBuffer };
    const normalized = normalizeLandmarks(smoothedMsg); latestFrameRef.current = normalized;
    const features = selectFeatures(normalized.vector);
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
    inferencePushRef.current(features); sentenceOnActivityRef.current(msg.hasLeftHand || msg.hasRightHand, performance.now());
  }, []);
  const { ready, error, fps, attachVideo, attachCanvas, setInferenceSize } = useHolisticPipeline(handleLandmarks);

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
      <StatusBar
        items={[
          { label: "Camera", ready, error },
          { label: "Model", ready: inference.ready, error: inference.error },
          { label: `Tier ${tierIndex + 1}/${RESOLUTION_TIERS.length}`, ready: true, tooltip: `Resolution tier ${tierIndex + 1} of ${RESOLUTION_TIERS.length}. Auto-downgrades when fps stays below ${FPS_DOWNGRADE_THRESHOLD} for ${LOW_FPS_CONSECUTIVE_SAMPLES}s.${tierInferenceSize ? " Inference downscale " + tierInferenceSize.width + "x" + tierInferenceSize.height + " (UNVALIDATED — ?downscale=0 to disable)" : " No inference downscale."}` },
        ]}
        fps={fps}
        offline={offline}
      />

      <main className="relative mx-auto max-w-6xl px-4 pt-12 pb-20 sm:px-6 sm:py-12">
        <div className="grid-pattern" aria-hidden="true" />
        <div className="ambient-glow" aria-hidden="true" />
        <div className="noise-overlay" aria-hidden="true" />
        <AnimatePresence mode="wait">
        {view === "landing" && (
          <motion.div
            key={LANDING_KEY}
            className="relative z-10 flex flex-col items-center gap-8 pt-8"
            variants={landingContainer}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.25 }}
          >
            <motion.p variants={landingItem} className="m-0 text-sm font-bold uppercase tracking-[0.18em] text-cyan-400">Filipino Sign Language</motion.p>
            <motion.h1
              variants={landingItem}
              className="mt-2 text-5xl font-black tracking-tight sm:text-6xl gradient-text-brand"
              style={{ filter: "drop-shadow(0 0 40px rgba(6,182,212,0.35)) drop-shadow(0 0 80px rgba(6,182,212,0.15))" }}
            >
              HandSon
            </motion.h1>
            <motion.p variants={landingItem} className="mt-4 text-lg leading-7 text-slate-300 max-w-xl text-center">
              Choose how you want to practice Filipino Sign Language.
            </motion.p>
            <motion.div variants={landingItem} className="grid gap-4 sm:grid-cols-2 mt-4 w-full max-w-2xl">
              {inference.ready ? (
                <motion.div variants={landingItem}>
                  <GlassCard onClick={() => setView("predict")}>
                    <p className="text-2xl font-black text-cyan-300">Predict Mode</p>
                    <p className="mt-2 text-sm text-slate-400">Live camera sign recognition — sign and see the prediction in real time</p>
                  </GlassCard>
                </motion.div>
              ) : (
                <motion.div
                  variants={landingItem}
                  className="rounded-xl border border-cyan-400/15 bg-cyan-400/[0.02] p-8 text-center opacity-70"
                >
                  <p className="text-2xl font-black text-cyan-300/60">Predict Mode</p>
                  {inference.error ? (
                    <p className="mt-2 text-sm text-red-400">Model failed to load — check console for details</p>
                  ) : (
                    <div className="mt-2 flex flex-col items-center gap-2">
                      <div className="flex size-8 items-center justify-center rounded-full border-2 border-white/10 border-t-cyan-400/60 animate-spin" aria-hidden="true" />
                      <p className="mt-1 text-sm text-slate-500">Loading FSL model — Predict Mode unavailable until ready</p>
                    </div>
                  )}
                </motion.div>
              )}
              <motion.div variants={landingItem}>
                <GlassCard onClick={() => setView("text")}>
                  <p className="text-2xl font-black text-white">Text & Avatar</p>
                  <p className="mt-2 text-sm text-slate-400">Type English and watch the 3D avatar translate it into sign</p>
                </GlassCard>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
        {view === "predict" && (
          <motion.div
            key={PREDICT_KEY}
            className={`grid gap-6 ${inference.ready ? "lg:grid-cols-[minmax(0,1.45fr)_minmax(19rem,0.8fr)]" : "grid-cols-1"} lg:items-start`}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
          >
            <motion.section
              className="space-y-4 bg-slate-900/40 border border-slate-800/80 rounded-2xl p-4 relative backdrop-blur-xl shadow-2xl"
              variants={sectionVariants}
              initial="initial"
              animate="animate"
              transition={sectionTransition(0.05)}
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setView("landing")}
                  className="rounded-lg px-3 py-1.5 text-sm font-bold text-slate-300 hover:bg-white/10 hover:text-white transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
                >
                  ← Back
                </button>
                <p className="text-sm font-bold text-cyan-400">Predict Mode</p>
              </div>
              <CameraPreview onVideoReady={attachVideo} onCanvasReady={attachCanvas} resolution={resolution} />
            </motion.section>
            <motion.aside
              className={`space-y-4 ${inference.ready ? "" : "lg:hidden"}`}
              variants={sectionVariants}
              initial="initial"
              animate="animate"
              transition={sectionTransition(0.1)}
            >
              <PredictionOverlay ready={inference.ready} error={inference.error} prediction={inference.prediction} />
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
            </motion.aside>
          </motion.div>
        )}
        {view === "text" && (
          <motion.div
            key={TEXT_KEY}
            className="grid gap-6 lg:grid-cols-[minmax(0,1.45fr)_minmax(19rem,0.8fr)] lg:items-start animate-soft-in"
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
          >
            <motion.section
              className="space-y-4 bg-slate-900/40 border border-slate-800/80 rounded-2xl p-4 relative backdrop-blur-xl shadow-2xl"
              variants={sectionVariants}
              initial="initial"
              animate="animate"
              transition={sectionTransition(0.05)}
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setView("landing")}
                  className="rounded-lg px-3 py-1.5 text-sm font-bold text-slate-300 hover:bg-white/10 hover:text-white transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
                >
                  ← Back
                </button>
                <p className="text-sm font-bold text-cyan-400">Text & Avatar</p>
              </div>
              {showAvatar && <SignAvatar signSequence={glossResult?.glosses.map(g => g.gloss)} />}
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
                  className={`rounded-lg px-3 py-2.5 text-sm font-bold transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-4 focus:ring-cyan-400/30 ${
                    showAvatar ? "bg-white text-cyan-800 shadow-sm" : "text-slate-300 hover:bg-slate-700/40"
                  }`}
                >
                  {showAvatar ? "Hide 3D Avatar" : "Show 3D Avatar"}
                </button>
              </div>
              <div className={`rounded-xl border px-4 py-3 transition-colors duration-200 ${
                textStatus === "ready" ? "border-green-400/40 bg-green-400/10" :
                textStatus === "loading" ? "border-yellow-400/40 bg-yellow-400/10" :
                textStatus === "error" ? "border-red-400/40 bg-red-400/10" :
                "border-cyan-400/20 bg-white/5"
              }`}>
                <div className="flex items-center gap-2">
                  <div className={`w-2.5 h-2.5 rounded-full ${
                    textStatus === "ready" ? "bg-green-400" :
                    textStatus === "loading" ? "bg-yellow-400 animate-pulse" :
                    textStatus === "error" ? "bg-red-400" :
                    "bg-cyan-400"
                  }`} />
                  <p className="m-0 text-sm font-bold text-slate-200">
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
        <div className="mt-6 flex justify-end">
          <AccessibilityControls
            textSize={textSize}
            onTextSize={setTextSize}
            highContrast={highContrast}
            onHighContrast={setHighContrast}
          />
        </div>
        </AnimatePresence>
      </main>
    </div>
  );
}
