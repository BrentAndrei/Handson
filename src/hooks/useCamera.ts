import { useEffect, useRef, useState, useCallback } from "react";

export type CameraStatus = "idle" | "requesting" | "ready" | "error";

interface UseCameraOptions {
  width?: number;
  height?: number;
  facingMode?: "user" | "environment";
}

interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement>;
  status: CameraStatus;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

/**
 * Manages the lifecycle of a getUserMedia() stream bound to a <video> element.
 *
 * Architectural notes:
 * - The MediaStream itself lives in a ref, never in React state. State updates
 *   trigger re-renders; a stream object changing identity on every permission
 *   grant would cause unnecessary re-renders of anything that reads it.
 * - `mountedRef` guards against setting state after unmount if the user
 *   navigates away while the permission prompt is still pending — a common
 *   source of "Can't perform a React state update on an unmounted component"
 *   warnings and, worse, orphaned camera streams.
 * - Cleanup explicitly stops every MediaStreamTrack. Just dropping the
 *   reference does NOT release the hardware/camera indicator light — you
 *   must call track.stop() on each track.
 */
export function useCamera(options: UseCameraOptions = {}): UseCameraReturn {
  const { width = 640, height = 480, facingMode = "user" } = options;

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mountedRef = useRef(true);

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (mountedRef.current) setStatus("idle");
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("getUserMedia is not supported in this browser.");
      setStatus("error");
      return;
    }

    // Release any stream already running before requesting a new one —
    // e.g. Retry after a transient error, or a caller changing resolution
    // on an already-live camera. Without this, the old MediaStreamTracks
    // are simply overwritten and never stopped: the camera hardware
    // indicator light stays on and the old stream keeps decoding frames
    // nobody reads.
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setStatus("requesting");
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: width },
          height: { ideal: height },
          facingMode,
          // Cap the source frame rate at the camera level so we're not
          // decoding 30/60fps frames just to throw 6/36 of them away
          // downstream. Cheaper than throttling after the fact.
          frameRate: { ideal: 24, max: 30 },
        },
        audio: false,
      });

      // Component unmounted while the permission prompt was open — discard
      // the stream immediately rather than attaching it to a dead ref.
      if (!mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus("ready");
    } catch (err) {
      if (!mountedRef.current) return;
      const message =
        err instanceof DOMException
          ? mapMediaError(err)
          : "Unknown camera error.";
      setError(message);
      setStatus("error");
    }
  }, [width, height, facingMode]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { videoRef, status, error, start, stop };
}

function mapMediaError(err: DOMException): string {
  switch (err.name) {
    case "NotAllowedError":
      return "Camera permission was denied.";
    case "NotFoundError":
      return "No camera device was found.";
    case "NotReadableError":
      return "Camera is already in use by another application.";
    case "OverconstrainedError":
      return "No camera satisfies the requested constraints.";
    default:
      return `Camera error: ${err.message}`;
  }
}
