import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BrutalButton } from "../components/ui/BrutalButton";
import { BrutalCard } from "../components/ui/BrutalCard";
import { BrutalBadge } from "../components/ui/BrutalBadge";
import { BrutalLabel } from "../components/ui/BrutalLabel";
import { IconCheck, IconSpark, IconVideoImport } from "../components/icons";
import { CameraPreview } from "../components/CameraPreview";
import SignAvatar from "../components/SignAvatar";
import { useHolisticPipeline } from "../hooks/useHolisticPipeline";
import { normalizeLandmarks } from "../utils/normalizeLandmarks";
import { selectFeatures, FEATURE_DIM } from "../features/featureSelect";
import { cn } from "../lib/cn";
import {
  CONFIDENCE_PRESETS,
  avatarDataFilename,
  buildAvatarDataJson,
  createVideoLandmarker,
  extractClip,
  mirrorFrame,
  type ClipCoverage,
  type ExtractedClip,
} from "../lib/videoCapture";
import {
  loadSignModel,
  testClipAgainstModel,
  type ClipTestResult,
} from "../lib/clipTester";
import { INFERENCE_LABELS } from "../inference/labels";
import { WINDOW_SIZE, DEFAULT_STRIDE, sliceIntoWindows } from "../features/windowBuffer";
import type { LandmarksResult } from "../mediapipe/types";
import type { AppView } from "../components/ui/BrutalNav";

/** The avatar-data loaders assume 30 fps. */
const FPS = 30;
/** Live capture runs at the pipeline's 24 fps — the same rate the
 *  shipped corpus was captured at. */
const RECORD_FPS = 24;
/** Hard cap so a forgotten REC cannot grow unbounded. */
const MAX_RECORD_SEC = 30;

type PendingClip = {
  id: number;
  label: string;
  startSec: number;
  endSec: number;
};

interface FinishedClip {
  id: number;
  label: string;
  frames: number[];
  frameCount: number;
  coverage: ClipCoverage;
  filename: string;
  source: "webcam" | "video";
  fps: number;
}

/** Per-clip classifier-test state, keyed by clip id. */
type ClipTestState =
  | { status: "running"; done: number; total: number }
  | { status: "done"; result: ClipTestResult }
  | { status: "error"; message: string };

function formatSec(s: number): string {
  return `${s.toFixed(2)}s`;
}

/** One detection-group coverage bar. Colour is never the only signal:
 *  the percentage is always printed next to the bar. */
function CoverageBar({
  label,
  value,
  total,
}: {
  label: string;
  value: number;
  total: number;
}) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 shrink-0 truncate text-right text-[10px] font-extrabold uppercase tracking-[0.12em] text-[var(--ink-soft)]">
        {label}
      </span>
      <div
        className="h-[10px] flex-1 overflow-hidden rounded-full border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)]"
        role="img"
        aria-label={`${label} present in ${pct} percent of frames`}
      >
        <div
          className={cn(
            "h-full",
            pct >= 90
              ? "bg-[var(--lime)]"
              : pct >= 50
                ? "bg-[var(--sun)]"
                : "bg-[var(--coral)]"
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-10 shrink-0 text-[11px] font-extrabold tabular-nums">
        {pct}%
      </span>
    </div>
  );
}

export function VideoCaptureScreen({
  onNavigate,
}: {
  onNavigate: (v: AppView) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const clipIdRef = useRef(1);
  const avatarBoxRef = useRef<HTMLDivElement | null>(null);

  const [tab, setTab] = useState<"import" | "record" | "train">("import");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const [labelInput, setLabelInput] = useState("");
  const [inMark, setInMark] = useState<number | null>(null);
  const [outMark, setOutMark] = useState<number | null>(null);
  const [clips, setClips] = useState<PendingClip[]>([]);
  const [preset, setPreset] = useState<keyof typeof CONFIDENCE_PRESETS>("balanced");

  const [extracting, setExtracting] = useState<{
    label: string;
    done: number;
    total: number;
  } | null>(null);
  const [results, setResults] = useState<FinishedClip[]>([]);
  const [tests, setTests] = useState<Map<number, ClipTestState>>(new Map());
  const [error, setError] = useState<string | null>(null);

/** Existing corpus labels → filename, so the UI can say whether a
    *  clip replaces a recording or introduces a new label. */
  const [manifestFiles, setManifestFiles] = useState<Map<string, string>>(
    new Map()
  );

  /** Training dataset state — built from captured clips. */
  const [trainingDataset, setTrainingDataset] = useState<{
    featureDim: number;
    windowSize: number;
    labels: string[];
    samples: { label: string; frames: number[] }[];
  } | null>(null);
  const [exportingDataset, setExportingDataset] = useState(false);

  useEffect(() => {
    fetch("/avatar-data/manifest.json")
      .then((r) => r.json())
      .then((m) => {
        const map = new Map<string, string>();
        const entries = Array.isArray(m?.entries) ? m.entries : [];
        for (const e of entries) {
          if (e?.label && e?.file) map.set(e.label, e.file);
        }
        setManifestFiles(map);
      })
      .catch(() => {});
  }, []);

  // ------------------------------------------------------------------
  // RECORD — webcam capture with a live avatar mimic.
  //
  // The studio runs its OWN HolisticLandmarker instance (the app's
  // pipeline belongs to the live views). Every inference tick:
  //   packed → normalizeLandmarks → selectFeatures
  // produces exactly the 345-float avatar-data frame, so what the
  // avatar shows while you sign IS what REC saves — no second pass,
  // no drift between preview and recording.
  // ------------------------------------------------------------------
  const mimicRef = useRef<{ vector: Float32Array; seq: number } | null>(null);
  const mimicSeqRef = useRef(0);
  const recordBufferRef = useRef<{
    frames: number[];
    coverage: ClipCoverage;
    startMs: number;
  } | null>(null);
  const recordCountRef = useRef(0);
  const [recording, setRecording] = useState(false);
  const [recordFrames, setRecordFrames] = useState(0);
  const [recordCoverage, setRecordCoverage] = useState<ClipCoverage>({
    frames: 0,
    pose: 0,
    leftHand: 0,
    rightHand: 0,
    face: 0,
  });

  const stopRecordingRef = useRef<() => void>(() => {});

  const handleStudioFrame = useCallback((msg: LandmarksResult) => {
    const normalized = normalizeLandmarks(msg);
    const features = selectFeatures(normalized.vector);
    mimicRef.current = { vector: features, seq: mimicSeqRef.current++ };

    const rec = recordBufferRef.current;
    if (!rec) return;
    rec.frames.push(...features);
    rec.coverage.frames++;
    if (msg.hasPose) rec.coverage.pose++;
    if (msg.hasLeftHand) rec.coverage.leftHand++;
    if (msg.hasRightHand) rec.coverage.rightHand++;
    if (msg.hasFace) rec.coverage.face++;

    recordCountRef.current++;
    // 4Hz UI update — a setState per inference tick would re-render
    // the whole tree 24x/second.
    if (recordCountRef.current % 6 === 0) {
      setRecordFrames(recordCountRef.current);
      setRecordCoverage({ ...rec.coverage });
    }
    if (performance.now() - rec.startMs >= MAX_RECORD_SEC * 1000) {
      stopRecordingRef.current();
    }
  }, []);

  const studioPipeline = useHolisticPipeline(handleStudioFrame, {
    confidence: CONFIDENCE_PRESETS[preset],
  });

  const startRecording = useCallback(() => {
    recordBufferRef.current = {
      frames: [],
      coverage: { frames: 0, pose: 0, leftHand: 0, rightHand: 0, face: 0 },
      startMs: performance.now(),
    };
    recordCountRef.current = 0;
    setRecordFrames(0);
    setRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    const buf = recordBufferRef.current;
    recordBufferRef.current = null;
    setRecording(false);
    setRecordFrames(recordCountRef.current);
    if (!buf || buf.frames.length < FEATURE_DIM) {
      setError("Nothing recorded — sign for at least one frame.");
      return;
    }
    const label = labelInput.trim().toUpperCase() || "UNTITLED";

    // The live pipeline detects at <= RECORD_FPS (24), but the
    // avatar-data convention — and the classifier's training — is
    // 30 fps, and playback duration is keyframeCount × 400 ms.
    // Hold-sample the detections to 30 fps by wall-clock so the
    // saved clip plays at the speed it was signed: output frame i
    // shows whichever detection was newest at time i/30 s.
    const elapsedSec = (performance.now() - buf.startMs) / 1000;
    const targetCount = Math.max(1, Math.round(elapsedSec * FPS));
    const detected = Math.floor(buf.frames.length / FEATURE_DIM);
    const frames = new Array(targetCount * FEATURE_DIM);
    for (let i = 0; i < targetCount; i++) {
      const src = Math.min(
        detected - 1,
        Math.floor((i * detected) / targetCount)
      );
      for (let j = 0; j < FEATURE_DIM; j++) {
        frames[i * FEATURE_DIM + j] = buf.frames[src * FEATURE_DIM + j];
      }
    }

    setResults((prev) => [
      ...prev,
      {
        id: clipIdRef.current++,
        label,
        frames,
        frameCount: targetCount,
        coverage: buf.coverage,
        filename: avatarDataFilename(label),
        source: "webcam",
        fps: FPS,
      },
    ]);
    setError(null);
    // Keep the mimic visible after REC stops: the last frame stays
    // published in mimicRef.
  }, [labelInput]);

  stopRecordingRef.current = stopRecording;

  // ------------------------------------------------------------------
  // PREVIEW — drive the shared avatar with a finished clip's frames.
  // ------------------------------------------------------------------
  const previewRef = useRef<{ vector: Float32Array; seq: number } | null>(null);
  const previewSeqRef = useRef(0);
  const previewIdxRef = useRef(0);
  const [previewId, setPreviewId] = useState<number | null>(null);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewSpeed, setPreviewSpeed] = useState(1);
  const [mirrored, setMirrored] = useState(false);
  const [exportingId, setExportingId] = useState<number | null>(null);
  /** Speed to restore after a WebM export (exports always run at 1×). */
  const preExportSpeedRef = useRef(1);

  useEffect(() => {
    previewIdxRef.current = 0;
    previewSeqRef.current = 0;
  }, [previewId]);

  useEffect(() => {
    if (!previewPlaying || previewId === null) return;
    const clip = results.find((r) => r.id === previewId);
    if (!clip) return;
    let raf = 0;
    let last = 0;
    const step = (now: number) => {
      const interval = 1000 / clip.fps / previewSpeed;
      if (now - last >= interval) {
        last = now;
        const i = previewIdxRef.current % clip.frameCount;
        const src = clip.frames.slice(
          i * FEATURE_DIM,
          (i + 1) * FEATURE_DIM
        );
        const vec = mirrored ? mirrorFrame(src) : new Float32Array(src);
        previewRef.current = { vector: vec, seq: previewSeqRef.current++ };
        previewIdxRef.current = i + 1;
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [previewPlaying, previewId, previewSpeed, mirrored, results]);

  /** Render the avatar signing one clip to a .webm. */
  const exportClip = useCallback(
    (clip: FinishedClip) => {
      const canvas = avatarBoxRef.current?.querySelector("canvas");
      if (!canvas) {
        setError("Preview not ready — play the clip once first.");
        return;
      }
      let recorder: MediaRecorder;
      try {
        const stream = canvas.captureStream(clip.fps);
        const mime =
          typeof MediaRecorder !== "undefined" &&
          MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
            ? "video/webm;codecs=vp9"
            : "video/webm";
        recorder = new MediaRecorder(stream, { mimeType: mime });
      } catch {
        setError("Video export is not supported in this browser.");
        return;
      }
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      const finished = new Promise<Blob>((resolve) => {
        recorder.onstop = () =>
          resolve(new Blob(chunks, { type: "video/webm" }));
      });
      recorder.start(250);
      setExportingId(clip.id);
      setPreviewId(clip.id);
      // Exports always run at natural speed: a slow-motion
      // preview would otherwise stretch the video 4×.
      preExportSpeedRef.current = previewSpeed;
      setPreviewSpeed(1);
      setPreviewPlaying(true);
      const durationMs = (clip.frameCount / clip.fps) * 1000;
      setTimeout(() => {
        setPreviewPlaying(false);
        setPreviewSpeed(preExportSpeedRef.current);
        recorder.stop();
        setExportingId(null);
      }, durationMs + 150);
      void finished.then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${clip.label.replace(/[^a-zA-Z0-9]/g, "_")}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      });
    },
    [previewSpeed]
  );

  /** Run the captured clip through the sign classifier and
   *  report whether the model reads it as the clip's label. */
  const runClipTest = useCallback(async (clip: FinishedClip) => {
    setTests((prev) => {
      const next = new Map(prev);
      next.set(clip.id, { status: "running", done: 0, total: 0 });
      return next;
    });
    try {
      const model = await loadSignModel();
      const result = await testClipAgainstModel(
        model,
        clip.frames,
        clip.label,
        INFERENCE_LABELS,
        (done, total) =>
          setTests((prev) => {
            const next = new Map(prev);
            next.set(clip.id, { status: "running", done, total });
            return next;
          })
      );
      setTests((prev) => {
        const next = new Map(prev);
        next.set(clip.id, { status: "done", result });
        return next;
      });
    } catch (e) {
      setTests((prev) => {
        const next = new Map(prev);
        next.set(clip.id, {
          status: "error",
          message: e instanceof Error ? e.message : "Test failed",
        });
        return next;
      });
    }
  }, []);

  const deleteClip = useCallback((id: number) => {
    setResults((prev) => prev.filter((r) => r.id !== id));
    setTests((prev) => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
    setPreviewId((cur) => (cur === id ? null : cur));
    setPreviewPlaying(false);
  }, []);

  // ------------------------------------------------------------------
  // IMPORT — video file handling.
  // ------------------------------------------------------------------
  const loadFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("video/")) {
        setError("That file is not a video — drop a video file (.mp4, .webm, .mov).");
        return;
      }
      setError(null);
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      setVideoFile(file);
      setDuration(0);
      setCurrentTime(0);
      setPlaying(false);
      setInMark(null);
      setOutMark(null);
      setLabelInput(file.name.replace(/\.[^.]+$/, "").toUpperCase());
    },
    [videoUrl]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) loadFile(file);
    },
    [loadFile]
  );

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (playing) {
      video.pause();
    } else {
      video.play().catch(() => {});
    }
    setPlaying(!playing);
  }, [playing]);

  const onTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (video) setCurrentTime(video.currentTime);
  }, []);

  const onScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    const t = Number(e.target.value);
    setCurrentTime(t);
    if (video) video.currentTime = t;
  }, []);

  const markIn = useCallback(() => setInMark(currentTime), [currentTime]);
  const markOut = useCallback(() => setOutMark(currentTime), [currentTime]);

  const addClip = useCallback(() => {
    if (inMark === null || outMark === null) {
      setError("Mark an in point and an out point first.");
      return;
    }
    const start = Math.min(inMark, outMark);
    const end = Math.max(inMark, outMark);
    if (end - start < 1 / FPS) {
      setError("Clip is shorter than one frame — widen the in/out range.");
      return;
    }
    const label = labelInput.trim().toUpperCase();
    if (!label) {
      setError("Give the clip a label (the sign's name).");
      return;
    }
    setError(null);
    setClips((prev) => [
      ...prev,
      { id: clipIdRef.current++, label, startSec: start, endSec: end },
    ]);
    setInMark(null);
    setOutMark(null);
  }, [inMark, outMark, labelInput]);

  const runExtraction = useCallback(async () => {
    const video = videoRef.current;
    if (!video || clips.length === 0) return;
    setError(null);
    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const landmarker = await createVideoLandmarker(CONFIDENCE_PRESETS[preset]);
      for (const clip of clips) {
        const extracted: ExtractedClip = await extractClip(
          video,
          landmarker,
          { startSec: clip.startSec, endSec: clip.endSec },
          clip.label,
          {
            fps: FPS,
            signal: abort.signal,
            onProgress: (done, total) =>
              setExtracting({ label: clip.label, done, total }),
          }
        );
        setResults((prev) => [
          ...prev,
          {
            id: clip.id,
            label: clip.label,
            frames: extracted.frames,
            frameCount: extracted.frameCount,
            coverage: extracted.coverage,
            filename: avatarDataFilename(clip.label),
            source: "video",
            fps: FPS,
          },
        ]);
      }
      landmarker.close();
      setClips([]);
    } catch (e) {
      if (!abort.signal.aborted) {
        setError(e instanceof Error ? e.message : "Extraction failed");
      }
    } finally {
      setExtracting(null);
      abortRef.current = null;
    }
  }, [clips, preset]);

  const cancelExtraction = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const download = useCallback((result: FinishedClip) => {
    const json = buildAvatarDataJson(result);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = result.filename;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const downloadAll = useCallback(() => {
    results.forEach((r) => download(r));
  }, [results, download]);

  /** Build a training dataset (fsl_dataset.json format) from captured clips.
    *  Each clip is sliced into sliding windows of WINDOW_SIZE frames with
    *  DEFAULT_STRIDE, exactly like the live pipeline and clipTester do. */
  const buildTrainingDataset = useCallback(() => {
    const samples: { label: string; frames: number[] }[] = [];
    for (const clip of results) {
      const frameCount = Math.floor(clip.frames.length / FEATURE_DIM);
      const frames: Float32Array[] = [];
      for (let i = 0; i < frameCount; i++) {
        frames.push(
          new Float32Array(
            clip.frames.slice(i * FEATURE_DIM, (i + 1) * FEATURE_DIM)
          )
        );
      }
      const windows = sliceIntoWindows(
        frames,
        FEATURE_DIM,
        WINDOW_SIZE,
        DEFAULT_STRIDE
      );
      for (const window of windows) {
        samples.push({ label: clip.label, frames: Array.from(window) });
      }
    }
    if (samples.length === 0) {
      setError("No clips to train on — capture or import some clips first.");
      return;
    }
    const labels = Array.from(new Set(samples.map((s) => s.label))).sort();
    setTrainingDataset({
      featureDim: FEATURE_DIM,
      windowSize: WINDOW_SIZE,
      labels,
      samples,
    });
    setError(null);
  }, [results]);

  const exportTrainingDataset = useCallback(() => {
    if (!trainingDataset) return;
    setExportingDataset(true);
    try {
      const json = JSON.stringify(trainingDataset, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "fsl_dataset.json";
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExportingDataset(false);
    }
  }, [trainingDataset]);

  const clipPlanned = (label: string) =>
    manifestFiles.has(label) ? "replaces" : "new";

  const previewClip = results.find((r) => r.id === previewId) ?? null;

  return (
    <div className="relative space-y-6">
      {/* PRINTED BACKDROP. Sun wash, halftone patch, three sparks —
          same visual language as the home screen. All pieces are
          aria-hidden, pointer-events-none, locked to -z-10 so they
          never interfere with interaction or widen the page. */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
        <span className="absolute -top-24 right-0 block h-[300px] w-[300px] rounded-full bg-[var(--sun)] sm:-top-28 sm:h-[360px] sm:w-[360px]" />
        <span className="hs-dots hs-rot-neg2 absolute -bottom-4 left-0 block h-[90px] w-[150px] opacity-30" />
        <IconSpark className="absolute left-[3px] top-[80px] h-7 w-7 text-[var(--coral)] hs-rot-3" />
        <IconSpark className="absolute right-[70px] top-[150px] h-5 w-5 text-[var(--teal)] hs-rot-neg3" />
        <IconSpark className="absolute bottom-[190px] right-[14px] h-6 w-6 text-[var(--lime)] hs-rot-1" />
      </div>

      {/* Registration marks, asymmetric on purpose. */}
      <span aria-hidden="true" className="hs-mark-cross absolute -right-1 top-16 hidden opacity-40 lg:block" />
      <span aria-hidden="true" className="hs-mark-cross absolute left-1/2 top-4 hidden opacity-30 lg:block" />

      {/* Header */}
      <div className="flex items-center gap-3">
        <BrutalButton tone="paper" size="sm" onClick={() => onNavigate("home")}>
          ← Back
        </BrutalButton>
        <div>
          <p className="m-0 text-[11px] font-extrabold uppercase tracking-[0.22em] text-[var(--teal-deep)]">
            Dataset tools
          </p>
          <h1 className="m-0 font-[family-name:var(--font-display)] text-3xl uppercase tracking-tight">
            Capture studio
          </h1>
        </div>
      </div>

      {/* Section rule with spark — editorial device tying header to the
          studio guide. */}
      <div className="mb-4 flex items-center gap-3">
        <BrutalLabel tone="mute">How it works</BrutalLabel>
        <span aria-hidden="true" className="h-[3px] flex-1 bg-[var(--ink)]" />
        <IconSpark aria-hidden className="h-4 w-4 shrink-0 text-[var(--coral)] hs-rot-3" />
      </div>

      {/* How it works */}
      <BrutalCard depth="lg" className="p-5">
        <ol className="m-0 list-none space-y-2 p-0 text-sm">
          <li>
            <strong>Record</strong> your own signs straight from the
            webcam — front-facing, no rig needed. The avatar mirrors you
            live, so what you see is exactly what gets saved.
          </li>
          <li>
            <strong>Import</strong> a tutorial video instead: drop the
            file, scrub to a sign, mark the in/out points, and extract.
            Videos are processed entirely in your browser — they never
            leave this device.
          </li>
          <li>
            <strong>Verify, then install.</strong> Every clip plays
            through the 3D avatar right here, with detection coverage
            printed per group — and you can{" "}
            <strong>test it against the sign classifier</strong> to
            see whether the model reads it correctly before it
            reaches the dataset.
          </li>
          <li>
            When it looks right, download the JSON into{" "}
            <code className="rounded border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-1">
              handson/public/avatar-data/
            </code>{" "}
            and restart the dev server.
          </li>
        </ol>
        <p className="m-0 mt-3 border-t-[var(--border-w-thin)] border-[var(--ink)] pt-3 text-[12px] font-bold text-[var(--ink-soft)]">
          The avatar replays the signer's hands — a generic model, not a
          mirror of you. Left-handed source? Flip it with the mirror
          toggle so the avatar signs with your dominant hand.
</p>
        </BrutalCard>

      {/* Section rule with spark — separates the guide from the tabs. */}
      <div className="mb-4 flex items-center gap-3">
        <BrutalLabel tone="mute">Capture source</BrutalLabel>
        <span aria-hidden="true" className="h-[3px] flex-1 bg-[var(--ink)]" />
        <IconSpark aria-hidden className="h-4 w-4 shrink-0 text-[var(--teal)] hs-rot-neg3" />
      </div>

      {/* Tab bar */}
      <div role="tablist" aria-label="Capture source" className="flex gap-2">
        {(["record", "import", "train"] as const).map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
            className={cn(
              "min-h-[44px] rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)] px-4 font-[family-name:var(--font-display)] text-sm uppercase tracking-[0.08em] transition-[transform,box-shadow,background-color] duration-[var(--press-ms)]",
              tab === t
                ? "bg-[var(--teal)] text-[var(--ink)] shadow-[var(--shadow-sm)]"
                : "bg-[var(--paper-sunken)] text-[var(--ink)] hover:bg-[var(--paper-raised)]"
            )}
          >
            {t === "record"
              ? "Record (webcam)"
              : t === "import"
                ? "Import (video file)"
                : "Train model"}
          </button>
        ))}
      </div>

      {tab === "record" ? (
        <motion.section
          key="record"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          aria-label="Record from webcam"
          className="space-y-4"
        >
          <BrutalCard depth="lg" className="p-5">
            <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <p className="mb-2 m-0 text-[10px] font-extrabold uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                    You
                  </p>
                  <CameraPreview
                    onVideoReady={studioPipeline.attachVideo}
                    onCanvasReady={studioPipeline.attachCanvas}
                    resolution={{ width: 960, height: 540 }}
                    showFlipButton
                  />
                </div>
                <div>
                  <p className="mb-2 m-0 text-[10px] font-extrabold uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                    Avatar mirror
                  </p>
                  <BrutalCard
                    surface="ink"
                    depth="md"
                    className="flex min-h-[240px] items-center justify-center lg:min-h-[338px]"
                  >
                    <div className="h-[240px] w-full lg:h-[338px]">
                      <SignAvatar
                        liveFrameRef={mimicRef}
                        liveActive={studioPipeline.ready}
                      />
                    </div>
                  </BrutalCard>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-end gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                    Sign label
                  </span>
                  <input
                    type="text"
                    value={labelInput}
                    onChange={(e) => setLabelInput(e.target.value)}
                    placeholder="e.g. KNOW"
                    className="min-h-[44px] w-40 rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)] bg-[var(--paper)] px-3 font-[family-name:var(--font-display)] text-sm uppercase tracking-[0.08em]"
                  />
                </label>
                {recording ? (
                  <BrutalButton tone="coral" size="md" onClick={stopRecording}>
                    ■ Stop — save clip
                  </BrutalButton>
                ) : (
                  <BrutalButton
                    tone="ink"
                    size="md"
                    onClick={startRecording}
                    disabled={!studioPipeline.ready}
                  >
                    ● Rec
                  </BrutalButton>
                )}
                {recording && (
                  <span
                    className="flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.1em] text-[var(--danger)]"
                    aria-live="polite"
                  >
                    <span
                      className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-[var(--danger)]"
                      aria-hidden="true"
                    />
                    Recording — {recordFrames} frames (
                    {formatSec(recordFrames / RECORD_FPS)})
                  </span>
                )}
              </div>

              {recording && (
                <div className="mt-3 grid max-w-md grid-cols-2 gap-x-4 gap-y-1.5">
                  <CoverageBar
                    label="Pose"
                    value={recordCoverage.pose}
                    total={Math.max(recordCoverage.frames, 1)}
                  />
                  <CoverageBar
                    label="Left hand"
                    value={recordCoverage.leftHand}
                    total={Math.max(recordCoverage.frames, 1)}
                  />
                  <CoverageBar
                    label="Right hand"
                    value={recordCoverage.rightHand}
                    total={Math.max(recordCoverage.frames, 1)}
                  />
                  <CoverageBar
                    label="Face"
                    value={recordCoverage.face}
                    total={Math.max(recordCoverage.frames, 1)}
                  />
                </div>
              )}

              <p className="m-0 mt-3 text-[12px] font-bold text-[var(--ink-soft)]">
                Face the camera straight on, shoulders and both hands in
                frame. The avatar copies you live at {RECORD_FPS} fps —
                the clip is resampled to the 30 fps avatar-data
                convention on save, so it plays back at the speed you
                signed it. Clips cap at {MAX_RECORD_SEC}s.
              </p>
</BrutalCard>
          </motion.section>
        ) : tab === "import" ? (
          <motion.section
            key="import"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            aria-label="Import from video file"
            className="space-y-4"
          >
            <BrutalCard depth="lg" className="p-5">
            <div
                role="button"
                tabIndex={0}
                aria-label="Drop a video file or press to choose one"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ")
                    fileInputRef.current?.click();
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                className={cn(
                  "flex min-h-[140px] cursor-pointer flex-col items-center justify-center gap-3 rounded-[var(--radius)] border-[var(--border-w)] border-dashed border-[var(--ink)] px-4 py-6 text-center transition-colors",
                  dragOver ? "bg-[var(--lime)]" : "bg-[var(--paper-sunken)]"
                )}
              >
                <IconVideoImport className="h-10 w-10" aria-hidden />
                <p className="m-0 text-sm font-extrabold uppercase tracking-[0.1em]">
                  {videoFile ? videoFile.name : "Drop a video here, or choose a file"}
                </p>
                <p className="m-0 text-[11px] font-bold text-[var(--ink-soft)]">
                  MP4 · WebM · MOV — processed locally
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="sr-only"
                aria-hidden
                tabIndex={-1}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) loadFile(file);
                  e.target.value = "";
                }}
              />

              {videoUrl && (
                <div className="mt-4">
                  <video
                    key={videoUrl}
                    ref={videoRef}
                    src={videoUrl}
                    muted
                    playsInline
                    onLoadedMetadata={(e) => {
                      setDuration(e.currentTarget.duration);
                      setCurrentTime(0);
                    }}
                    onTimeUpdate={onTimeUpdate}
                    onPlay={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onEnded={() => setPlaying(false)}
                    className="w-full rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)] bg-black"
                  />
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <BrutalButton tone="paper" size="sm" onClick={togglePlay}>
                      {playing ? "Pause" : "Play"}
                    </BrutalButton>
                    <input
                      type="range"
                      min={0}
                      max={duration || 0}
                      step={1 / FPS}
                      value={currentTime}
                      onChange={onScrub}
                      aria-label="Scrub timeline"
                      className="min-w-[180px] flex-1 accent-[var(--teal-deep)]"
                    />
                    <span className="text-[11px] font-extrabold tabular-nums">
                      {formatSec(currentTime)} / {formatSec(duration)}
                    </span>
                  </div>
                </div>
              )}
            </BrutalCard>

            <BrutalCard depth="lg" className="p-5">
              <h2 className="m-0 font-[family-name:var(--font-display)] text-lg uppercase tracking-tight">
                Clip markers
              </h2>
              <div className="mt-3 flex flex-wrap items-end gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                    Sign label
                  </span>
                  <input
                    type="text"
                    value={labelInput}
                    onChange={(e) => setLabelInput(e.target.value)}
                    placeholder="e.g. KNOW"
                    className="min-h-[44px] rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)] bg-[var(--paper)] px-3 font-[family-name:var(--font-display)] text-sm uppercase tracking-[0.08em]"
                  />
                </label>
                <BrutalButton tone="sun" size="sm" onClick={markIn} disabled={!videoUrl}>
                  Mark in {inMark !== null ? formatSec(inMark) : ""}
                </BrutalButton>
                <BrutalButton tone="sun" size="sm" onClick={markOut} disabled={!videoUrl}>
                  Mark out {outMark !== null ? formatSec(outMark) : ""}
                </BrutalButton>
                <BrutalButton tone="teal" size="sm" onClick={addClip}>
                  Add clip
                </BrutalButton>
              </div>

              {clips.length > 0 && (
                <ul className="mt-4 space-y-2">
                  {clips.map((clip) => (
                    <li
                      key={clip.id}
                      className="flex items-center justify-between gap-3 rounded-[var(--radius)] border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-3 py-2"
                    >
                      <span className="font-[family-name:var(--font-display)] text-sm font-bold uppercase tracking-[0.08em]">
                        {clip.label}
                      </span>
                      <span className="text-[11px] font-bold tabular-nums text-[var(--ink-soft)]">
                        {formatSec(clip.startSec)} → {formatSec(clip.endSec)} (
                        {(clip.endSec - clip.startSec).toFixed(2)}s)
                      </span>
                      <button
                        type="button"
                        onClick={() =>
                          setClips((prev) => prev.filter((c) => c.id !== clip.id))
                        }
                        aria-label={`Remove clip ${clip.label}`}
                        className="px-2 text-[11px] font-extrabold uppercase text-[var(--danger)] underline underline-offset-2"
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="mt-4 flex flex-wrap items-end gap-3 border-t-[var(--border-w-thin)] border-[var(--ink)] pt-4">
                <label className="flex flex-col gap-1">
                  <span className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                    Detection confidence
                  </span>
                  <select
                    value={preset}
                    onChange={(e) =>
                      setPreset(e.target.value as keyof typeof CONFIDENCE_PRESETS)
                    }
                    className="min-h-[44px] rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)] bg-[var(--paper)] px-3 text-sm font-bold"
                  >
                    <option value="permissive">Permissive — catches more hands</option>
                    <option value="balanced">Balanced (default)</option>
                    <option value="strict">Strict — cleanest landmarks</option>
                  </select>
                </label>
                {extracting ? (
                  <BrutalButton tone="coral" size="md" onClick={cancelExtraction}>
                    Cancel
                  </BrutalButton>
                ) : (
                  <BrutalButton
                    tone="teal"
                    size="md"
                    onClick={runExtraction}
                    disabled={clips.length === 0}
                    withArrow
                  >
                    Extract {clips.length > 0 ? `${clips.length} clip${clips.length > 1 ? "s" : ""}` : ""}
                  </BrutalButton>
                )}
              </div>

<AnimatePresence>
                {extracting ? (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mt-4"
                    aria-live="polite"
                  >
                    <p className="m-0 mb-2 text-[11px] font-extrabold uppercase tracking-[0.12em]">
                      Extracting "{extracting.label}" — {extracting.done}/
                      {extracting.total} frames
                    </p>
                    <div className="h-[14px] overflow-hidden rounded-full border-[var(--border-w)] border-[var(--ink)] bg-[var(--paper-sunken)]">
                      <div
                        className="h-full bg-[var(--teal)] transition-[width] duration-150"
                        style={{
                          width: `${Math.round((extracting.done / Math.max(extracting.total, 1)) * 100)}%`,
                        }}
                      />
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </BrutalCard>
          </motion.section>
        ) : (
          <motion.section
            key="train"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            aria-label="Train model"
            className="space-y-4"
          >
            <BrutalCard depth="lg" className="p-5">
              <div className="flex items-center justify-between gap-3 mb-4">
                <h2 className="m-0 font-[family-name:var(--font-display)] text-lg uppercase tracking-tight">
                  Train model
                </h2>
                {trainingDataset ? (
                  <BrutalBadge tone="lime">Dataset ready — {trainingDataset.samples.length} windows</BrutalBadge>
                ) : (
                  <BrutalBadge tone="paper">No dataset built</BrutalBadge>
                )}
              </div>

              <p className="mb-4 text-sm text-[var(--ink-soft)]">
                Build a training dataset from your captured clips and export it
                as <code className="rounded border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-1">fsl_dataset.json</code>.
                Run the Python training script locally, then convert the model to
                TensorFlow.js and replace <code className="rounded border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-1">handson/public/models/fsl_model/</code>.
              </p>

              {results.length === 0 ? (
                <p className="text-center py-8 text-[var(--ink-soft)]">
                  No captured clips yet. Record or import clips first, then come
                  back here to build the training dataset.
                </p>
              ) : (
                <>
                  <div className="space-y-2 mb-4">
                    <h3 className="m-0 text-[10px] font-extrabold uppercase tracking-[0.12em] text-[var(--ink-soft)]">
                      Captured clips available for training
                    </h3>
                    <ul className="space-y-1.5 max-h-60 overflow-y-auto">
                      {results.map((result) => {
                        const frameCount = Math.floor(
                          result.frames.length / FEATURE_DIM
                        );
                        const windows = Math.max(
                          0,
                          Math.floor(
                            (frameCount - WINDOW_SIZE) / DEFAULT_STRIDE
                          ) + 1
                        );
                        return (
                          <li
                            key={result.id}
                            className="flex items-center justify-between gap-3 rounded-[var(--radius)] border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-3 py-2"
                          >
                            <div className="flex flex-col gap-0.5">
                              <span className="font-[family-name:var(--font-display)] text-sm font-bold uppercase tracking-[0.08em]">
                                {result.label}
                              </span>
                              <span className="text-[10px] font-bold text-[var(--ink-soft)]">
                                {result.frameCount} frames @ {result.fps} fps →{" "}
                                {windows} window{windows !== 1 ? "s" : ""}
                              </span>
                            </div>
                            <BrutalBadge
                              tone={result.source === "webcam" ? "teal" : "sun"}
                            >
                              {result.source === "webcam" ? "webcam" : "video"}
                            </BrutalBadge>
                          </li>
                        );
                      })}
                    </ul>
                  </div>

                  {trainingDataset ? (
                    <div className="flex flex-wrap items-center gap-3 border-t-[var(--border-w-thin)] border-[var(--ink)] pt-4">
                      <BrutalButton
                        tone="ink"
                        size="sm"
                        onClick={exportTrainingDataset}
                        disabled={exportingDataset}
                        withArrow
                      >
                        {exportingDataset ? "Exporting…" : "Export fsl_dataset.json"}
                      </BrutalButton>
                      <BrutalButton
                        tone="paper"
                        size="sm"
                        onClick={() => setTrainingDataset(null)}
                      >
                        Clear dataset
                      </BrutalButton>
                    </div>
                  ) : (
                    <div className="border-t-[var(--border-w-thin)] border-[var(--ink)] pt-4">
                      <BrutalButton
                        tone="teal"
                        size="md"
                        onClick={buildTrainingDataset}
                        withArrow
                      >
                        Build training dataset ({results.length} clip{results.length > 1 ? "s" : ""})
                      </BrutalButton>
                    </div>
                  )}

                  {trainingDataset && (
                    <details className="mt-4">
                      <summary className="cursor-pointer text-[11px] font-extrabold uppercase tracking-[0.1em] text-[var(--ink-soft)]">
                        Dataset details (JSON structure)
                      </summary>
                      <pre className="mt-2 p-3 rounded-[var(--radius)] bg-[var(--paper-sunken)] text-[10px] overflow-x-auto max-h-64">
                        {JSON.stringify(
                          {
                            featureDim: trainingDataset.featureDim,
                            windowSize: trainingDataset.windowSize,
                            labels: trainingDataset.labels,
                            sampleCount: trainingDataset.samples.length,
                          },
                          null,
                          2
                        )}
                      </pre>
                    </details>
                  )}
                </>
              )}
            </BrutalCard>

            <BrutalCard depth="lg" className="p-5">
              <h3 className="m-0 mb-3 font-[family-name:var(--font-display)] text-base uppercase tracking-tight">
                Training workflow
              </h3>
              <ol className="m-0 list-decimal list-inside space-y-2 p-0 text-sm text-[var(--ink-soft)]">
                <li>
                  Click <strong>Build training dataset</strong> above to slice all
                  captured clips into {WINDOW_SIZE}-frame windows (stride {DEFAULT_STRIDE}).
                </li>
                <li>
                  Click <strong>Export fsl_dataset.json</strong> to download the
                  dataset file.
                </li>
                <li>
                  Run the Python training script:
                  <pre className="mt-1 ml-4 p-2 rounded-[var(--radius)] bg-[var(--paper-sunken)] text-[10px] overflow-x-auto">
{`python training/train.py \\
  --dataset fsl_dataset.json \\
  --out ./model_out \\
  --epochs 200 \\
  --batch-size 8`}
                  </pre>
                </li>
                <li>
                  Convert the trained Keras model to TensorFlow.js:
                  <pre className="mt-1 ml-4 p-2 rounded-[var(--radius)] bg-[var(--paper-sunken)] text-[10px] overflow-x-auto">
{`tensorflowjs_converter \\
  --input_format=tf_saved_model \\
  --output_format=tfjs_layers_model \\
  ./model_out/saved_model \\
  ./model_out/tfjs_model`}
                  </pre>
                </li>
                <li>
                  Replace the contents of{" "}
                  <code className="rounded border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-1">
                    handson/public/models/fsl_model/
                  </code>{" "}
                  with the generated TFJS model files, then restart the dev
                  server.
                </li>
              </ol>
            </BrutalCard>
          </motion.section>
        )}

      {error && (
        <p
          role="alert"
          className="rounded-[var(--radius)] border-[var(--border-w)] border-[var(--danger)] bg-[var(--paper-sunken)] px-3 py-2 text-sm font-bold text-[var(--danger)]"
        >
          {error}
        </p>
      )}

      {/* Section rule with spark — editorial cap before the clips list. */}
      <div className="mb-4 flex items-center gap-3">
        <BrutalLabel tone="mute">Captured clips</BrutalLabel>
        <span aria-hidden="true" className="h-[3px] flex-1 bg-[var(--ink)]" />
        <IconSpark aria-hidden className="h-4 w-4 shrink-0 text-[var(--sun)] hs-rot-1" />
      </div>

      {/* Results + avatar preview */}
      <AnimatePresence>
        {results.length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            aria-label="Captured clips"
            className="space-y-4"
          >
            <div className="flex items-center justify-between gap-3">
              <h2 className="m-0 font-[family-name:var(--font-display)] text-lg uppercase tracking-tight">
                Captured clips
              </h2>
              <BrutalButton tone="lime" size="sm" onClick={downloadAll}>
                Download all
              </BrutalButton>
            </div>

            {/* The avatar preview player. The wrapper div (not the
                card) carries the ref: BrutalCard does not forward
                refs, and the WebM export needs the canvas inside. */}
            <div ref={avatarBoxRef}>
              <BrutalCard
                surface="ink"
                depth="lg"
                className="relative overflow-hidden p-4"
              >
              <div className="pointer-events-none absolute left-3 top-3 z-10">
                <BrutalBadge tone="teal">
                  {previewClip ? previewClip.label : "Avatar preview"}
                </BrutalBadge>
              </div>
              <div className="h-[280px] w-full sm:h-[340px]">
                <SignAvatar
                  liveFrameRef={previewRef}
                  liveActive={previewId !== null}
                />
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <BrutalButton
                  tone={previewPlaying ? "coral" : "lime"}
                  size="sm"
                  onClick={() => setPreviewPlaying((p) => !p)}
                  disabled={!previewClip}
                >
                  {previewPlaying ? "Pause" : "Play"}
                </BrutalButton>
                <label className="flex items-center gap-1 text-[11px] font-extrabold uppercase tracking-[0.1em] text-[var(--paper)]">
                  Speed
                  <select
                    value={previewSpeed}
                    onChange={(e) => setPreviewSpeed(Number(e.target.value))}
                    className="rounded-[var(--radius)] border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-raised)] px-2 py-1 text-[var(--ink)]"
                    aria-label="Playback speed"
                  >
                    {[0.25, 0.5, 1, 2].map((s) => (
                      <option key={s} value={s}>
                        {s}×
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  onClick={() => setMirrored((m) => !m)}
                  aria-pressed={mirrored}
                  className={cn(
                    "min-h-[38px] rounded-[var(--radius)] border-[var(--border-w)] border-[var(--ink)] px-3 text-[11px] font-extrabold uppercase tracking-[0.08em] transition-colors",
                    mirrored
                      ? "bg-[var(--pink)] text-[var(--ink)]"
                      : "bg-[var(--paper-raised)] text-[var(--ink)] hover:bg-[var(--paper-sunken)]"
                  )}
                >
                  Mirror handedness
                </button>
                {previewClip && (
                  <span className="text-[10px] font-bold text-[var(--paper)]">
                    Frame {previewIdxRef.current % Math.max(previewClip.frameCount, 1)} /{" "}
                    {previewClip.frameCount}
                  </span>
                )}
              </div>
            </BrutalCard>
            </div>

            {results.map((result) => {
              const plan = clipPlanned(result.label);
              const isPreview = previewId === result.id;
              const test = tests.get(result.id);
              return (
                <BrutalCard key={result.id} depth="md" className="p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="flex items-center gap-2 font-[family-name:var(--font-display)] text-base font-bold uppercase tracking-[0.08em]">
                      {result.label}
                      <BrutalBadge tone={result.source === "webcam" ? "teal" : "sun"}>
                        {result.source === "webcam" ? "webcam" : "video"}
                      </BrutalBadge>
                    </span>
                    <span className="text-[11px] font-bold tabular-nums text-[var(--ink-soft)]">
                      {result.frameCount} frames @ {result.fps} fps
                    </span>
                  </div>
                  <div className="mt-3 space-y-1.5">
                    <CoverageBar
                      label="Pose"
                      value={result.coverage.pose}
                      total={result.coverage.frames}
                    />
                    <CoverageBar
                      label="Left hand"
                      value={result.coverage.leftHand}
                      total={result.coverage.frames}
                    />
                    <CoverageBar
                      label="Right hand"
                      value={result.coverage.rightHand}
                      total={result.coverage.frames}
                    />
                    <CoverageBar
                      label="Face"
                      value={result.coverage.face}
                      total={result.coverage.frames}
                    />
                  </div>

                  {test?.status === "running" && (
                    <div className="mt-3" aria-live="polite">
                      <p className="m-0 mb-2 text-[11px] font-extrabold uppercase tracking-[0.12em]">
                        Testing against the classifier
                        {test.total > 0
                          ? ` — ${test.done}/${test.total} windows`
                          : ""}
                      </p>
                      <div className="h-[14px] overflow-hidden rounded-full border-[var(--border-w)] border-[var(--ink)] bg-[var(--paper-sunken)]">
                        <div
                          className="h-full bg-[var(--sun)] transition-[width] duration-150"
                          style={{
                            width: `${Math.round((test.done / Math.max(test.total, 1)) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                  {test?.status === "done" && (
                    <div className="mt-3">
                      <div className="flex flex-wrap items-center gap-2">
{test.result.agree ? (
                          <BrutalBadge tone="lime">
                            Model agrees — reads "{result.label}"
                          </BrutalBadge>
                        ) : (
                          <BrutalBadge tone="coral">
                            Model reads it as "{test.result.top[0]?.label}"
                          </BrutalBadge>
                        )}
                        <span className="text-[10px] font-bold text-[var(--ink-soft)]">
                          {test.result.windows} windows
                        </span>
                      </div>
                      <div className="mt-2 space-y-1.5">
                        {test.result.top.map((p) => (
                          <CoverageBar
                            key={p.label}
                            label={p.label}
                            value={p.confidence}
                            total={1}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                  {test?.status === "error" && (
                    <p
                      role="alert"
                      className="mt-2 text-[11px] font-bold text-[var(--danger)]"
                    >
                      {test.message}
                    </p>
                  )}

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t-[var(--border-w-thin)] border-[var(--ink)] pt-3">
                    <code className="rounded border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-2 py-1 text-[11px] font-bold">
                      {result.filename}
                    </code>
                    <div className="flex flex-wrap items-center gap-2">
                      {plan === "replaces" ? (
                        <span className="flex items-center gap-1 text-[10px] font-extrabold uppercase tracking-[0.1em] text-[var(--teal-deep)]">
                          <IconCheck className="h-3.5 w-3.5" aria-hidden />
                          Replaces existing recording
                        </span>
                      ) : (
                        <span className="text-[10px] font-extrabold uppercase tracking-[0.1em] text-[var(--ink-soft)]">
                          New label — add “{result.label}” to the labels array in manifest.json
                        </span>
                      )}
                      <BrutalButton
                        tone={isPreview ? "coral" : "paper"}
                        size="sm"
                        onClick={() => {
                          setPreviewId(result.id);
                          setPreviewPlaying(true);
                        }}
                      >
                        {isPreview ? "Previewing" : "Preview"}
                      </BrutalButton>
                      <BrutalButton
                        tone="sun"
                        size="sm"
                        onClick={() => runClipTest(result)}
                        disabled={test?.status === "running"}
                      >
                        {test?.status === "running" ? "Testing…" : "Test"}
                      </BrutalButton>
                      <BrutalButton
                        tone="ink"
                        size="sm"
                        onClick={() => download(result)}
                      >
                        JSON
                      </BrutalButton>
                      <BrutalButton
                        tone="purple"
                        size="sm"
                        onClick={() => exportClip(result)}
                        disabled={exportingId !== null}
                      >
                        {exportingId === result.id ? "Exporting…" : "WebM"}
                      </BrutalButton>
                      <button
                        type="button"
                        onClick={() => deleteClip(result.id)}
                        aria-label={`Delete clip ${result.label}`}
                        className="px-2 text-[11px] font-extrabold uppercase text-[var(--danger)] underline underline-offset-2"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </BrutalCard>
              );
            })}
          </motion.section>
        )}
      </AnimatePresence>

      {/* INK FOOTER BAND. Thin printed rule capped by a registration
          mark — the page's colophon. Keeps the printed-paper metaphor
          all the way to the bottom edge. */}
      <div aria-hidden="true" className="mt-10 flex items-center justify-end gap-3">
        <span className="h-[1px] flex-1 bg-[var(--ink)] opacity-30" />
        <span className="hs-mark-cross text-[var(--ink)] opacity-40" />
      </div>
    </div>
  );
}
