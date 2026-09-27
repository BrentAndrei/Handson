import { useCallback, useEffect, useRef, useState } from "react";
import { FilesetResolver, HolisticLandmarker, HolisticLandmarkerResult, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import { packHolisticResult } from "../mediapipe/pack";
import { drawHolisticResult } from "../mediapipe/drawResult";
import { HolisticSmoother } from "../mediapipe/smoothing";
import type { SmoothedHolisticResult } from "../mediapipe/smoothing";
import type { LandmarksResult } from "../mediapipe/types";

const TARGET_FPS = 24;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS; // ~41.667ms

const MEDIAPIPE_VERSION = "0.10.20";
const MEDIAPIPE_WASM_BASE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

interface PipelineState {
  ready: boolean;
  error: string | null;
  fps: number;
}

/**
 * Linearly extrapolates one landmark group forward using velocity from the
 * previous detection. Pure function — no refs, no side effects.
 */
function extrapolateGroup(
  curr: NormalizedLandmark[] | undefined,
  prev: NormalizedLandmark[] | undefined,
  dtMs: number,
  elapsedMs: number
): NormalizedLandmark[] {
  if (!curr || curr.length === 0) return curr || [];
  if (!prev || prev.length === 0) return curr;

  const factor = Math.min(elapsedMs / dtMs, 1.5);

  return curr.map((p, i) => ({
    x: p.x + (p.x - prev[i].x) * factor,
    y: p.y + (p.y - prev[i].y) * factor,
    z: p.z + (p.z - prev[i].z) * factor,
    visibility: p.visibility,
  }));
}

export function useHolisticPipeline(onLandmarks: (result: LandmarksResult) => void) {
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const accumulatorRef = useRef(0);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const canvasElRef = useRef<HTMLCanvasElement | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const fpsCounterRef = useRef({ count: 0, windowStart: 0 });
  const smootherRef = useRef<HolisticSmoother | null>(null);

  const onLandmarksRef = useRef(onLandmarks);
  useEffect(() => {
    onLandmarksRef.current = onLandmarks;
  }, [onLandmarks]);

  const [state, setState] = useState<PipelineState>({
    ready: false,
    error: null,
    fps: 0,
  });

  const prevRawRef = useRef<HolisticLandmarkerResult | null>(null);
  const currRawRef = useRef<HolisticLandmarkerResult | null>(null);
  const prevTsRef = useRef<number>(-1);
  const currTsRef = useRef<number>(-1);

  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inferenceSizeRef = useRef<{ width: number; height: number } | null>(null);

  const getInferenceSource = useCallback((video: HTMLVideoElement): HTMLVideoElement | HTMLCanvasElement => {
    const size = inferenceSizeRef.current;
    if (!size) return video;

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    const canvas = offscreenCanvasRef.current;
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width;
      canvas.height = size.height;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return video;

    ctx.drawImage(video, 0, 0, size.width, size.height);
    return canvas;
  }, []);

  const getDisplaySmoothed = useCallback((now: number): SmoothedHolisticResult => {
    const curr = currRawRef.current;
    const currTs = currTsRef.current;
    const prev = prevRawRef.current;
    const prevTs = prevTsRef.current;

    if (!curr || currTs < 0) {
      return {
        pose: { landmarks: null, opacity: 0 },
        leftHand: { landmarks: null, opacity: 0 },
        rightHand: { landmarks: null, opacity: 0 },
        face: { landmarks: null, opacity: 0 },
      };
    }

    const elapsed = now - currTs;
    const MAX_EXTRAPOLATION_MS = 150;

    if (elapsed > MAX_EXTRAPOLATION_MS || !prev || prevTs < 0) {
      if (!smootherRef.current) smootherRef.current = new HolisticSmoother();
      return smootherRef.current.update(curr, currTs);
    }

    const dt = currTs - prevTs;
    if (dt <= 0) {
      if (!smootherRef.current) smootherRef.current = new HolisticSmoother();
      return smootherRef.current.update(curr, currTs);
    }

    const extrapolated = {
      poseLandmarks: [extrapolateGroup(curr.poseLandmarks[0], prev.poseLandmarks[0], dt, elapsed)],
      leftHandLandmarks: [extrapolateGroup(curr.leftHandLandmarks[0], prev.leftHandLandmarks[0], dt, elapsed)],
      rightHandLandmarks: [extrapolateGroup(curr.rightHandLandmarks[0], prev.rightHandLandmarks[0], dt, elapsed)],
      faceLandmarks: [extrapolateGroup(curr.faceLandmarks[0], prev.faceLandmarks[0], dt, elapsed)],
    };

    if (!smootherRef.current) smootherRef.current = new HolisticSmoother();
    return smootherRef.current.updateExtrapolated(extrapolated, now);
  }, []);

  const setInferenceSize = useCallback((size: { width: number; height: number } | null) => {
    inferenceSizeRef.current = size;
  }, []);

  // --- Landmarker setup ----------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE_URL);

        const landmarker = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          minPoseDetectionConfidence: 0.7,
          minPosePresenceConfidence: 0.7,
          minFaceDetectionConfidence: 0.7,
          minFacePresenceConfidence: 0.7,
          minHandLandmarksConfidence: 0.7,
        });

        if (cancelled) {
          landmarker.close();
          return;
        }

        landmarkerRef.current = landmarker;
        setState((s) => ({ ...s, ready: true }));
      } catch (err) {
        if (cancelled) return;
        setState((s) => ({
          ...s,
          error:
            err instanceof Error ? err.message : "Failed to initialize HolisticLandmarker",
        }));
      }
    })();

    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, []);

  // --- Capture + detect loop ------------------------------------------------
  //
  // One loop, one rAF. Detection and display are two phases of the same
  // frame so the overlay can never drift out of sync with the detector's
  // state — and so a frame where detection doesn't fire still redraws the
  // last smoothed result (or its hold/fade) instead of leaving a stale
  // canvas behind. A separate display-only loop would race the detector's
  // own rAF and paint over it mid-frame.
  const loop = useCallback((now: number) => {
    if (lastFrameTimeRef.current === 0) lastFrameTimeRef.current = now;
    const delta = now - lastFrameTimeRef.current;
    lastFrameTimeRef.current = now;

    accumulatorRef.current = Math.min(
      accumulatorRef.current + delta,
      FRAME_INTERVAL_MS * 2
    );

    const landmarker = landmarkerRef.current;
    const video = videoElRef.current;

    if (
      accumulatorRef.current >= FRAME_INTERVAL_MS &&
      landmarker &&
      video &&
      video.readyState >= 2 &&
      video.currentTime !== lastVideoTimeRef.current
    ) {
      accumulatorRef.current -= FRAME_INTERVAL_MS;
      lastVideoTimeRef.current = video.currentTime;

      try {
        const source = getInferenceSource(video);
        const ts = performance.now();
        const result = landmarker.detectForVideo(source, ts);

        if (!smootherRef.current) smootherRef.current = new HolisticSmoother();
        const smoothed = smootherRef.current.update(result, ts);
        drawOverlay(video, canvasElRef, smoothed);

        prevRawRef.current = currRawRef.current;
        currRawRef.current = result;
        prevTsRef.current = currTsRef.current;
        currTsRef.current = ts;

        const packed = packHolisticResult(result, ts);
        trackFps(fpsCounterRef, (fps) =>
          setState((s) => (s.fps === fps ? s : { ...s, fps }))
        );
        onLandmarksRef.current(packed);
      } catch (err) {
        setState((s) => ({
          ...s,
          error: err instanceof Error ? err.message : "Detection failed",
        }));
      }
    }

    // Display phase — runs every frame, independent of whether detection
    // fired. This is what keeps the overlay alive across the frames between
    // detections: the smoother holds the last pose and fades it, so the
    // canvas keeps showing a (dimming) skeleton instead of going blank.
    const videoForDraw = videoElRef.current;
    if (videoForDraw && smootherRef.current) {
      const display = getDisplaySmoothed(performance.now());
      drawOverlay(videoForDraw, canvasElRef, display);
    }

    rafRef.current = requestAnimationFrame(loop);
  }, [getInferenceSource, getDisplaySmoothed]);

  const attachVideo = useCallback(
    (video: HTMLVideoElement) => {
      videoElRef.current = video;
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(loop);
      }
    },
    [loop]
  );

const attachCanvas = useCallback((canvas: HTMLCanvasElement) => {
    canvasElRef.current = canvas;
  }, []);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  return { ...state, attachVideo, attachCanvas, setInferenceSize };
}

/**
 * Keeps the overlay canvas's drawing-buffer resolution in sync with the
 * video's intrinsic size, clears the previous frame, and draws the new one.
 */
function drawOverlay(
  video: HTMLVideoElement,
  canvasElRef: React.MutableRefObject<HTMLCanvasElement | null>,
  smoothed: SmoothedHolisticResult
) {
  const canvas = canvasElRef.current;
  if (!canvas || video.videoWidth === 0 || video.videoHeight === 0) return;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawHolisticResult(ctx, smoothed);
}

function trackFps(
  ref: React.MutableRefObject<{ count: number; windowStart: number }>,
  report: (fps: number) => void
) {
  const now = performance.now();
  if (ref.current.windowStart === 0) ref.current.windowStart = now;
  ref.current.count++;
  const elapsed = now - ref.current.windowStart;
  if (elapsed >= 1000) {
    report(Math.round((ref.current.count * 1000) / elapsed));
    ref.current.count = 0;
    ref.current.windowStart = now;
  }
}
