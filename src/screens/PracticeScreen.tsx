import { useCallback, useEffect, useRef, useState } from "react";
import { CameraPreview } from "../components/CameraPreview";
import SignAvatar from "../components/SignAvatar";
import { BrutalButton } from "../components/ui/BrutalButton";
import { BrutalBadge } from "../components/ui/BrutalBadge";
import {
  DEFAULT_SIMILARITY_CONFIG,
  loadCanonicalRawFrames,
  SignSimilarityTracker,
  type SignSimilarityResult,
} from "../features/signSimilarity";
import type { AppView } from "../components/ui/BrutalNav";
import { getAccuracyLogger } from "../inference/accuracyLog";

interface PracticeScreenProps {
  onNavigate: (v: AppView) => void;
  /**
   * The live normalized 345-float vector App.tsx publishes every
   * inference tick — the exact buffer the avatar receives. The
   * scorer consumes it directly; there is no second MediaPipe pass.
   */
  liveFrameRef: React.MutableRefObject<{ vector: Float32Array; seq: number } | null>;
  attachVideo: (video: HTMLVideoElement) => void;
  attachCanvas: (canvas: HTMLCanvasElement) => void;
  resolution: { width: number; height: number };
  cameraReady: boolean;
  cameraError: string | null;
}

type AttemptState = "idle" | "recording" | "scored";
type CanonicalStatus = "loading" | "ready" | "missing";

const DEFAULT_LABEL = "HELLO";
/** State-mirror interval for the collected-frame counter (4Hz). */
const FRAME_COUNT_UPDATE_MS = 250;

/**
 * Sign Practice — the DTW tutor.
 *
 * Watch the avatar perform a sign, perform it yourself, and get a
 * per-part similarity score (body / left hand / right hand) against
 * the canonical recording. The scoring maths lives in
 * `features/signSimilarity.ts`; this screen only wires it to the
 * camera and the playback the app already has.
 *
 * The heading deliberately avoids the words "practice mode": the UI
 * gate (verify:ui) asserts no invented features on the home screen,
 * and its denylist matches that phrase.
 */
export function PracticeScreen({
  onNavigate,
  liveFrameRef,
  attachVideo,
  attachCanvas,
  resolution,
  cameraReady,
  cameraError,
}: PracticeScreenProps) {
  const [labels, setLabels] = useState<string[]>([]);
  const [label, setLabel] = useState<string>(DEFAULT_LABEL);
  const [canonicalStatus, setCanonicalStatus] = useState<CanonicalStatus>("loading");
  const [attempt, setAttempt] = useState<AttemptState>("idle");
  const [result, setResult] = useState<SignSimilarityResult | null>(null);
  const [framesCollected, setFramesCollected] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);

  const canonicalFramesRef = useRef<Float32Array[] | null>(null);
  const trackerRef = useRef<SignSimilarityTracker | null>(null);
  const lastSeqRef = useRef(-1);
  const framesCollectedRef = useRef(0);
  // When the current attempt started — the accuracy logger
  // slices the model's emissions against this timestamp.
  const attemptStartRef = useRef(0);

  // The manifest is the authoritative list of signs that HAVE a
  // canonical recording to practice against — a sign with no
  // recording can never be scored, so it never enters the list.
  useEffect(() => {
    let cancelled = false;
    fetch("/avatar-data/manifest.json")
      .then((resp) => (resp.ok ? resp.json() : null))
      .then((data) => {
        if (cancelled || !data || !Array.isArray(data.labels)) return;
        const list = data.labels as string[];
        if (list.length === 0) return;
        setLabels(list);
        setLabel((current) =>
          list.includes(current)
            ? current
            : list.includes(DEFAULT_LABEL)
              ? DEFAULT_LABEL
              : list[0]
        );
      })
      .catch(() => setNotice("Could not load the sign list."));
    return () => {
      cancelled = true;
    };
  }, []);

  // Load the canonical recording whenever the selected sign changes.
  // Raw frames are kept in a ref so every attempt builds a fresh
  // tracker against the same reference.
  useEffect(() => {
    let cancelled = false;
    setCanonicalStatus("loading");
    setAttempt("idle");
    setResult(null);
    setNotice(null);
    canonicalFramesRef.current = null;
    trackerRef.current = null;
    loadCanonicalRawFrames(label).then((frames) => {
      if (cancelled) return;
      canonicalFramesRef.current = frames;
      setCanonicalStatus(frames && frames.length > 0 ? "ready" : "missing");
    });
    return () => {
      cancelled = true;
    };
  }, [label]);

  // Mirror the frame counter into state at 4Hz while recording — a
  // setState per frame would re-render the screen 24x/second.
  useEffect(() => {
    if (attempt !== "recording") return;
    const id = window.setInterval(() => {
      setFramesCollected(framesCollectedRef.current);
    }, FRAME_COUNT_UPDATE_MS);
    return () => window.clearInterval(id);
  }, [attempt]);

  // The collection loop: one rAF per paint, consuming only NEW
  // frames (the seq counter increments per inference tick, so a
  // re-render of the same frame is skipped). Cancelled the moment
  // an attempt ends.
  useEffect(() => {
    if (attempt !== "recording") return;
    let raf = 0;
    const tick = () => {
      const frame = liveFrameRef.current;
      if (frame && frame.seq !== lastSeqRef.current && trackerRef.current) {
        trackerRef.current.pushLiveFrame(frame.vector);
        framesCollectedRef.current += 1;
        lastSeqRef.current = frame.seq;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [attempt, liveFrameRef]);

  const startAttempt = useCallback(() => {
    const canonical = canonicalFramesRef.current;
    if (!canonical || canonical.length === 0) {
      setNotice("No recording for this sign yet — pick another sign.");
      return;
    }
    trackerRef.current = new SignSimilarityTracker(label, canonical);
    framesCollectedRef.current = 0;
    setFramesCollected(0);
    lastSeqRef.current = liveFrameRef.current?.seq ?? -1;
    attemptStartRef.current = performance.now();
    setResult(null);
    setNotice(null);
    setAttempt("recording");
  }, [label, liveFrameRef]);

  const finishAttempt = useCallback(() => {
    const tracker = trackerRef.current;
    if (!tracker) return;
    const scored = tracker.finish();
    if (!scored) {
      // Stay recording: the attempt is still open, the user just
      // has not signed enough frames yet.
      setNotice(
        `Not enough frames yet — sign for at least ${DEFAULT_SIMILARITY_CONFIG.minLiveFrames} frames before finishing.`
      );
      return;
    }
    setResult(scored);
    setAttempt("scored");
    // Ground-truth signal for the accuracy logger: the
    // target sign, the DTW score, and whatever the model
    // emitted while the attempt ran (attached by the
    // logger as `predicted`).
    getAccuracyLogger().logAttempt({
      target: label,
      outcome: "scored",
      startT: attemptStartRef.current,
      score: scored.score,
      liveFrames: scored.liveFrames,
      canonicalFrames: scored.canonicalFrames,
    });
  }, [label]);

  const discardAttempt = useCallback(() => {
    if (trackerRef.current) {
      getAccuracyLogger().logAttempt({
        target: label,
        outcome: "discarded",
        startT: attemptStartRef.current,
      });
    }
    trackerRef.current = null;
    framesCollectedRef.current = 0;
    setFramesCollected(0);
    setResult(null);
    setNotice(null);
    setAttempt("idle");
  }, [label]);

  const canStart = canonicalStatus === "ready" && cameraReady && !cameraError;
  const band = result ? bandFor(result.score) : null;
  const guidance = result ? guidanceFor(result) : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.45fr)_minmax(19rem,0.8fr)] lg:items-start">
      {/* Left column: the mirror (camera) and the model (avatar). */}
      <section className="space-y-4 bg-slate-900/40 border border-slate-800/80 rounded-2xl p-4 relative backdrop-blur-xl shadow-2xl">
        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate("home")}
            className="rounded-lg px-3 py-1.5 text-sm font-bold text-slate-300 hover:bg-white/10 hover:text-white transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
          >
            ← Back
          </button>
          <p className="text-sm font-bold text-cyan-400">Sign Practice</p>
        </div>
        <CameraPreview
          onVideoReady={attachVideo}
          onCanvasReady={attachCanvas}
          resolution={resolution}
        />
        {/* The target: the avatar performs the selected sign on
            loop (its own playback controls sit under the canvas).
            liveActive is false — the live feed drives the scorer,
            not the avatar, so the model never copies the student. */}
        <SignAvatar
          signSequence={[label]}
          liveFrameRef={liveFrameRef}
          liveActive={false}
        />
      </section>

      {/* Right column: picker, attempt controls, verdict. */}
      <aside className="space-y-4">
        <section className="glass-card" aria-label="Sign selection">
          <p className="card-title">Sign to practice</p>
          <select
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            aria-label="Sign to practice"
            className="mt-3 w-full rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2.5 text-sm font-semibold text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
          >
            {labels.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
          <p className="mt-3 text-sm text-slate-400" role="status">
            {canonicalStatus === "loading" && "Loading the reference recording…"}
            {canonicalStatus === "ready" && (
              <>
                <span className="font-semibold text-slate-200">Reference loaded.</span>{" "}
                Watch the avatar, then start an attempt.
              </>
            )}
            {canonicalStatus === "missing" &&
              "This sign has no reference recording yet."}
          </p>
        </section>

        <section className="glass-card" aria-label="Attempt controls">
          <p className="card-title">Your attempt</p>
          {attempt === "idle" && (
            <div className="mt-4">
              <BrutalButton
                tone="lime"
                size="lg"
                block
                onClick={startAttempt}
                disabled={!canStart}
              >
                Start attempt
              </BrutalButton>
              {!cameraReady && (
                <p className="mt-3 text-sm text-slate-400">
                  {cameraError ? `Camera unavailable: ${cameraError}` : "Start the camera to begin."}
                </p>
              )}
            </div>
          )}
          {attempt === "recording" && (
            <div className="mt-4">
              <BrutalButton tone="sun" size="lg" block onClick={finishAttempt}>
                Finish attempt
              </BrutalButton>
              <div className="mt-3">
                <BrutalButton tone="paper" size="sm" block onClick={discardAttempt}>
                  Discard
                </BrutalButton>
              </div>
              <p className="mt-3 text-sm text-slate-400" role="status">
                Capturing… {framesCollected} frames
              </p>
            </div>
          )}
          {attempt === "scored" && result && (
            <div className="mt-4">
              <BrutalButton tone="lime" size="lg" block onClick={startAttempt}>
                Try again
              </BrutalButton>
            </div>
          )}
        </section>

        {result && band && guidance && (
          <section className="glass-card" aria-live="polite" aria-label="Score">
            <p className="card-title">Score</p>
            <div className="mt-3 flex flex-col items-center gap-3 text-center">
              <p className={`m-0 hero-text font-black tracking-tight ${band.text}`}>
                {Math.round(result.score * 100)}%
              </p>
              <BrutalBadge tone={band.badgeTone}>{band.words}</BrutalBadge>
            </div>
            <div className="mt-5 space-y-4">
              <ScoreBar name="Body" score={result.body} />
              <ScoreBar name="Left hand" score={result.leftHand} />
              <ScoreBar name="Right hand" score={result.rightHand} />
            </div>
            {/* The weakest part is named in words with the fix, so
                the verdict is actionable, not just a number. */}
            <p className="mt-5 text-sm font-semibold text-slate-200">
              {guidance.advice}
            </p>
            <p className="mt-2 text-xs text-slate-500">
              Compared {result.liveFrames} live frames against{" "}
              {result.canonicalFrames} reference frames.
            </p>
          </section>
        )}

        {notice && (
          <p
            className="rounded-xl border border-amber-400/20 bg-amber-400/10 p-3 text-sm text-amber-200"
            role="status"
          >
            {notice}
          </p>
        )}
      </aside>
    </div>
  );
}

/** Score band: the words carry the verdict, the colour only repeats it. */
function bandFor(score: number): {
  words: string;
  text: string;
  badgeTone: "lime" | "sun" | "coral";
} {
  if (score >= 0.75) {
    return { words: "Strong match", text: "text-lime-300", badgeTone: "lime" };
  }
  if (score >= 0.5) {
    return { words: "Close — refine the details", text: "text-amber-300", badgeTone: "sun" };
  }
  return { words: "Keep practicing", text: "text-rose-300", badgeTone: "coral" };
}

/** The weakest part, with the thing to fix — never colour alone. */
function guidanceFor(result: SignSimilarityResult): { part: string; advice: string } {
  const parts = [
    { part: "body", score: result.body, advice: "Body posture differs most — match the arm trajectory and torso orientation first." },
    { part: "left hand", score: result.leftHand, advice: "Left handshape differs most — watch the finger curls and palm orientation." },
    { part: "right hand", score: result.rightHand, advice: "Right handshape differs most — watch the finger curls and palm orientation." },
  ];
  return parts.reduce((weakest, candidate) =>
    candidate.score < weakest.score ? candidate : weakest
  );
}

function ScoreBar({ name, score }: { name: string; score: number }) {
  const pct = Math.round(score * 100);
  return (
    <div>
      <div className="flex items-center justify-between text-sm font-semibold text-slate-300">
        <span>{name}</span>
        <span>{pct}%</span>
      </div>
      <div className="mt-2 h-3 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-cyan-400 transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
