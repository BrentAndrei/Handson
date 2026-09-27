import { useEffect, useRef, useState } from "react";
import { useCamera } from "../hooks/useCamera";
import { CameraSkeleton } from "./Skeleton";

interface CameraPreviewProps {
  onVideoReady?: (video: HTMLVideoElement) => void;
  onCanvasReady?: (canvas: HTMLCanvasElement) => void;
  resolution?: { width: number; height: number };
  className?: string;
}

const DEFAULT_RESOLUTION = { width: 960, height: 540 };

/**
 * The camera panel — the focal element of Predict mode.
 *
 * The webcam feed fills a dark, rounded frame with a subtle glass rim. The
 * landmark overlay (pose/hands/face) is drawn on a <canvas> stacked above the
 * <video>, never behind a glass pane — the skeleton colors stay readable
 * against the dark feed, and the frame itself stays undistorted.
 *
 * Glass styling belongs to the chrome AROUND the feed (this frame's rim, the
 * status bar, the control panels), not on the video.
 */
export function CameraPreview({ onVideoReady, onCanvasReady, resolution = DEFAULT_RESOLUTION, className }: CameraPreviewProps) {
  const { videoRef, status, error, start, stop } = useCamera({ width: resolution.width, height: resolution.height });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [aspectRatio, setAspectRatio] = useState(`${resolution.width} / ${resolution.height}`);
  useEffect(() => { start(); }, [resolution.width, resolution.height]);
  useEffect(() => () => stop(), []);
  useEffect(() => { if (status === "ready" && videoRef.current) onVideoReady?.(videoRef.current); }, [status, onVideoReady]);
  // Attach the canvas ref when the canvas actually appears. The canvas is
  // only rendered once the camera is ready, so an effect that runs once on
  // mount would see canvasRef.current === null and never attach — leaving
  // the pipeline's canvasElRef null for the whole session (the overlay then
  // silently no-ops and the frame looks empty). Watching `status` re-runs
  // this the moment the <canvas> mounts.
  useEffect(() => {
    if (status === "ready" && canvasRef.current) {
      onCanvasReady?.(canvasRef.current);
    }
  }, [status, onCanvasReady]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const sync = () => { if (video.videoWidth && video.videoHeight) setAspectRatio(`${video.videoWidth} / ${video.videoHeight}`); };
    sync(); video.addEventListener("loadedmetadata", sync);
    return () => video.removeEventListener("loadedmetadata", sync);
  }, [status]);
return (
    <div
      className={`glass relative w-full overflow-hidden border border-slate-800/80 ${className ?? ""}`}
      style={{ aspectRatio }}
    >
      {status === "ready" ? (
        <>
          <video ref={videoRef} muted playsInline className="absolute inset-0 size-full -scale-x-100 object-cover" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full -scale-x-100 object-cover" />
        </>
      ) : status === "error" ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-slate-950/80 p-5 text-center text-base text-white">
          <div>{error}</div>
          <button
            onClick={start}
            className="rounded-lg bg-white/10 px-4 py-2 font-semibold text-white shadow-sm backdrop-blur-sm transition-all duration-200 active:scale-[0.98] hover:bg-white/20 focus:outline-none focus:ring-4 focus:ring-white/20"
          >
            Retry
          </button>
        </div>
      ) : (
        <CameraSkeleton />
      )}
    </div>
  );
}
