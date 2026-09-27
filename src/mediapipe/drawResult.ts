import { HolisticLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { SmoothedHolisticResult } from "./smoothing";

// @mediapipe/tasks-vision ships a `DrawingUtils` class in its TypeScript
// declarations, but this package version's actual runtime bundle
// (vision_bundle.mjs) never exports it as an ES module — it's only wired up
// as a legacy global via an internal `goog.exportSymbol`-style call, so
// `import { DrawingUtils } from "@mediapipe/tasks-vision"` fails at build
// time with "is not exported by ... vision_bundle.mjs" even though the
// types say it should work. Landmark connection data (e.g.
// HolisticLandmarker.POSE_CONNECTIONS) IS genuinely exported — only the
// drawing helper class isn't — so this file draws with plain Canvas 2D
// calls instead of depending on the broken export.

/** Mirrors the (non-exported) `Connection` shape from the package's types. */
interface Connection {
  start: number;
  end: number;
}

interface Style {
  color: string;
  lineWidth: number;
  dotRadius?: number;
  dotFillColor?: string;
}

const POSE_STYLE: Style = { color: "#00E5FF", lineWidth: 3, dotRadius: 3, dotFillColor: "#003C42" };
const LEFT_HAND_STYLE: Style = { color: "#7CFF6B", lineWidth: 2, dotRadius: 2, dotFillColor: "#173B12" };
const RIGHT_HAND_STYLE: Style = { color: "#FFB020", lineWidth: 2, dotRadius: 2, dotFillColor: "#402D08" };
// Face gets a thin, low-opacity contour rather than the full ~800-edge
// tesselation — visually noisy and unnecessary at 24fps for a status
// overlay; contours (oval + eyes + brows + lips) are enough to confirm
// tracking is alive without drowning the frame. No dots for face — 132
// contour points' worth of circles would be visual clutter.
const FACE_STYLE: Style = { color: "rgba(255,255,255,0.5)", lineWidth: 1 };

// BlazePose's 33-point topology includes 11 face-region points (nose,
// eyes, ears, mouth corners — indices 0-10) alongside the body skeleton,
// so pose connects to the head at all. Drawing those here produces a
// second, much coarser face wireframe layered directly on top of the
// dedicated 468-point face mesh above — visually indistinguishable from a
// genuinely broken/misaligned overlay even though both are individually
// correct. Only pose's body connections (shoulders and below) add
// anything the face mesh doesn't already cover, so that's all we draw.
const POSE_FACE_LANDMARK_COUNT = 11;
const POSE_BODY_CONNECTIONS: Connection[] = HolisticLandmarker.POSE_CONNECTIONS.filter(
  ({ start, end }) => start >= POSE_FACE_LANDMARK_COUNT && end >= POSE_FACE_LANDMARK_COUNT
);

// Bridge joints: pose, the two hands, and the face are four independently
// detected groups (different crops, different models, different per-frame
// confidence), so their landmarks never perfectly coincide — there's
// always a small natural gap between e.g. pose's wrist estimate and the
// dedicated hand model's wrist estimate. Drawing one connecting line per
// gap turns "four separate floating clusters" into a single continuous
// skeleton without pretending the two wrist estimates are the same point.
const POSE_LEFT_SHOULDER = 11;
const POSE_RIGHT_SHOULDER = 12;
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;
// Chin (bottom of the 468-point face oval) is the face mesh's stable point
// closest to the neck — the one deliberate exception to "pose doesn't draw
// into the face" above: a single joining line, not pose's whole rough
// face sub-skeleton.
const FACE_CHIN_INDEX = 152;
const NECK_BRIDGE_STYLE: Style = { color: "#00E5FF", lineWidth: 2 };

function drawConnectors(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  connections: Connection[],
  style: Style,
  opacity: number
): void {
  if (opacity <= 0) return;
  const { width, height } = ctx.canvas;
  ctx.save();
  ctx.globalAlpha *= opacity;
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.lineWidth;
  ctx.beginPath();
  for (const { start, end } of connections) {
    const a = landmarks[start];
    const b = landmarks[end];
    if (!a || !b) continue;
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
  }
  ctx.stroke();
  ctx.restore();
}

function drawLandmarkDots(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  style: Style,
  opacity: number,
  minIndex = 0
): void {
  if (opacity <= 0) return;
  const { width, height } = ctx.canvas;
  const radius = style.dotRadius ?? 3;
  ctx.save();
  ctx.globalAlpha *= opacity;
  ctx.fillStyle = style.dotFillColor ?? style.color;
  ctx.strokeStyle = style.color;
  ctx.lineWidth = 1;
  for (let i = minIndex; i < landmarks.length; i++) {
    const p = landmarks[i];
    ctx.beginPath();
    ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function midpoint(a: NormalizedLandmark, b: NormalizedLandmark): NormalizedLandmark {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 } as NormalizedLandmark;
}

/**
 * Draws a single line between two points that may come from entirely
 * different landmark groups (e.g. pose's wrist and the hand model's
 * wrist). `opacity` should be the min of both groups' own opacities, so a
 * bridge never renders pointing at a group that isn't actually shown this
 * frame — e.g. pose visible but a hand that's genuinely left the frame and
 * finished its own fade-out.
 */
function drawBridge(
  ctx: CanvasRenderingContext2D,
  a: NormalizedLandmark | undefined,
  b: NormalizedLandmark | undefined,
  style: Style,
  opacity: number
): void {
  if (!a || !b || opacity <= 0) return;
  const { width, height } = ctx.canvas;
  ctx.save();
  ctx.globalAlpha *= opacity;
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.lineWidth;
  ctx.beginPath();
  ctx.moveTo(a.x * width, a.y * height);
  ctx.lineTo(b.x * width, b.y * height);
  ctx.stroke();
  ctx.restore();
}

/**
 * Renders one frame's holistic landmarks onto `ctx` from already-smoothed
 * groups (see mediapipe/smoothing.ts). Each group carries its own opacity:
 * 1 for a live detection this frame, fading toward 0 while a brief dropout
 * is being held over, and the group is simply absent once it's genuinely
 * gone. Caller is responsible for clearing the canvas beforehand.
 */
export function drawHolisticResult(
  ctx: CanvasRenderingContext2D,
  smoothed: SmoothedHolisticResult
): void {
  const { pose, leftHand, rightHand, face } = smoothed;

  // Optional diagnostic badge — shows which groups are currently visible,
  // so a teacher can tell at a glance whether the detector is working (vs.
  // an empty overlay that looks identical to "no detection"). Off by default;
  // flip `window.__drawDebug__ = true` in the console to enable.
  if ((window as any).__drawDebug__) {
    const groups = [
      ["pose", pose],
      ["L", leftHand],
      ["R", rightHand],
      ["face", face],
    ] as const;
    const active = groups
      .filter(([, g]) => g.landmarks && g.opacity > 0)
      .map(([n]) => n)
      .join(" ");
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(8, 8, 180, 26);
    ctx.fillStyle = "#00E5FF";
    ctx.font = "bold 13px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(active || "none", 14, 25);
    ctx.restore();
  }

  // Bridges first, so each group's own joints/dots layer cleanly on top of
  // the connecting lines rather than the lines drawing over them.
  if (pose.landmarks) {
    if (leftHand.landmarks) {
      drawBridge(
        ctx,
        pose.landmarks[POSE_LEFT_WRIST],
        leftHand.landmarks[0],
        LEFT_HAND_STYLE,
        Math.min(pose.opacity, leftHand.opacity)
      );
    }
    if (rightHand.landmarks) {
      drawBridge(
        ctx,
        pose.landmarks[POSE_RIGHT_WRIST],
        rightHand.landmarks[0],
        RIGHT_HAND_STYLE,
        Math.min(pose.opacity, rightHand.opacity)
      );
    }
    if (face.landmarks) {
      const neck = midpoint(pose.landmarks[POSE_LEFT_SHOULDER], pose.landmarks[POSE_RIGHT_SHOULDER]);
      drawBridge(
        ctx,
        neck,
        face.landmarks[FACE_CHIN_INDEX],
        NECK_BRIDGE_STYLE,
        Math.min(pose.opacity, face.opacity)
      );
    }
  }

  if (pose.landmarks) {
    drawConnectors(ctx, pose.landmarks, POSE_BODY_CONNECTIONS, POSE_STYLE, pose.opacity);
    drawLandmarkDots(ctx, pose.landmarks, POSE_STYLE, pose.opacity, POSE_FACE_LANDMARK_COUNT);
  }

if (face.landmarks) {
    drawConnectors(ctx, face.landmarks, HolisticLandmarker.FACE_LANDMARKS_CONTOURS, FACE_STYLE, face.opacity);
  }

  if (leftHand.landmarks) {
    drawConnectors(ctx, leftHand.landmarks, HolisticLandmarker.HAND_CONNECTIONS, LEFT_HAND_STYLE, leftHand.opacity);
    drawLandmarkDots(ctx, leftHand.landmarks, LEFT_HAND_STYLE, leftHand.opacity);
  }

  if (rightHand.landmarks) {
    drawConnectors(ctx, rightHand.landmarks, HolisticLandmarker.HAND_CONNECTIONS, RIGHT_HAND_STYLE, rightHand.opacity);
    drawLandmarkDots(ctx, rightHand.landmarks, RIGHT_HAND_STYLE, rightHand.opacity);
  }
}
