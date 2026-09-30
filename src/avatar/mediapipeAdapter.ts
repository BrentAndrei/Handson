/**
 * MediaPipe Holistic → CanonicalPose adapter.
 *
 * Converts the 345-float packed landmark buffer (33 pose + 21×2 hands +
 * 468 face) into the intermediate CanonicalPose representation.
 *
 * Coordinate handling:
 * - Input: MediaPipe normalized landmarks (shoulder-midpoint anchored,
 *   shoulder-width scaled). MediaPipe convention: X=right, Y=down, Z=inward.
 * - Output: Canonical space (X=forward, Y=up, Z=right, meters).
 * - The normalization scale (shoulder width) is preserved so that
 *   avatar bone directions can be computed directly from landmark
 *   directions.
 *
 * MediaPipe Pose landmark indices (33):
 *   0  = nose
 *   1-3 = eyes
 *   7  = left ear
 *   8  = right ear
 *   9-10 = mouth
 *   11 = left shoulder    12 = right shoulder
 *   13 = left elbow       14 = right elbow
 *   15 = left wrist        16 = right wrist
 *   17 = left pinky MCP    18 = right pinky MCP
 *   19 = left index MCP    20 = right index MCP
 *   21 = left thumb MCP    22 = right thumb MCP
 *   23 = left hip          24 = right hip
 *   25 = left knee         26 = right knee
 *   27 = left ankle        28 = right ankle
 *   29 = left heel         30 = right heel
 *   31 = left foot idx     32 = right foot idx
 *
 * MediaPipe Hand landmark indices (21 per hand):
 *   0  = wrist
 *   1-4  = thumb (MCP, PIP, DIP, TIP)
 *   5-8  = index (MCP, PIP, DIP, TIP)
 *   9-12 = middle (MCP, PIP, DIP, TIP)
 *   13-16 = ring (MCP, PIP, DIP, TIP)
 *   17-20 = pinky (MCP, PIP, DIP, TIP)
 */
import {
  HAND_COUNT,
  POSE_OFFSET,
  LEFT_HAND_OFFSET,
  RIGHT_HAND_OFFSET,
} from "../mediapipe/types";
import {
  mediapipeToCanonical,
} from "./coordinateSystem";
import {
  CanonicalPose,
  BodyPose,
  HandPose,
  FingerPose,
  FingerName,
  Vec3,
} from "./canonicalPose";

const VEC3_POOL: Vec3[] = [];
function vec3(x: number, y: number, z: number): Vec3 {
  const v = VEC3_POOL.pop() ?? [0, 0, 0];
  v[0] = x; v[1] = y; v[2] = z;
  return v;
}
function freeVec(v: Vec3) {
  if (VEC3_POOL.length < 300) VEC3_POOL.push(v);
}

function add(a: Vec3, b: Vec3): Vec3 {
  return vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
function scale(a: Vec3, s: number): Vec3 {
  return vec3(a[0] * s, a[1] * s, a[2] * s);
}
function len(a: Vec3): number {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
function normalize(a: Vec3): Vec3 {
  const l = len(a);
  if (l < 1e-10) return [0, 0, 0];
  return [a[0] / l, a[1] / l, a[2] / l];
}
function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** Read a landmark from the packed buffer as a Vec3 (raw MediaPipe coords). */
function readPoseLM(data: number[] | Float32Array, index: number): Vec3 {
  const base = POSE_OFFSET + index * 3;
  return vec3(data[base], data[base + 1], data[base + 2]);
}

/** Read a hand landmark from the packed buffer as a Vec3. */
function readHandLM(
  data: number[] | Float32Array,
  groupOffset: number,
  index: number
): Vec3 {
  const base = groupOffset + index * 3;
  return vec3(data[base], data[base + 1], data[base + 2]);
}

/**
 * MediaPipe Hand landmark indices.
 */
const HAND = {
  WRIST: 0,
  THUMB_MCP: 1, THUMB_PIP: 2, THUMB_DIP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

/**
 * MediaPipe Pose landmark indices.
 */
const POSE = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

/**
 * Convert a raw MediaPipe landmark to canonical space.
 * MediaPipe: X=right, Y=down, Z=inward.
 * Canonical: X=forward, Y=up, Z=right.
 */
function mpToCanon(v: Vec3): Vec3 {
  return mediapipeToCanonical(v);
}

/**
 * Compute the finger pose (direction + curl) from MediaPipe hand landmarks.
 *
 * @param wristWristPos The root of the hand (already in canonical space)
 * @param landmarks 21 hand landmarks in canonical space
 * @param finger Which finger
 */
function computeFingerPose(
  landmarks: Vec3[],
  finger: FingerName
): FingerPose {
  // HAND keys are ALL-CAPS ("THUMB_MCP" ...). The old
  // `finger.charAt(0).toUpperCase() + finger.slice(1)` produced "Thumb_MCP"
  // -> always undefined -> landmarks[undefined] -> crash on first finger.
  // (Path was dormant: mediapipeToCanonicalPose was never called until the
  // E8 handedness harness wired it up.)
  const fingerUpper = finger.toUpperCase();
  const mcpIdx = HAND[`${fingerUpper}_MCP` as keyof typeof HAND];
  const pipIdx = HAND[`${fingerUpper}_PIP` as keyof typeof HAND];
  const tipIdx = HAND[`${fingerUpper}_TIP` as keyof typeof HAND];
  const dipIdx = HAND[`${fingerUpper}_DIP` as keyof typeof HAND];

  const mcp = landmarks[mcpIdx];
  const pip = landmarks[pipIdx];
  const dip = landmarks[dipIdx];
  const tip = landmarks[tipIdx];

  // Finger direction: from MCP toward the tip
  const dirToTip = normalize(sub(tip, mcp));

  // Curl: angle at PIP joint between MCP→PIP and PIP→DIP
  // When straight: vectors are parallel, curl = 0
  // When curled: vectors diverge, curl approaches 1
  const mcpToPip = normalize(sub(pip, mcp));
  const pipToDip = normalize(sub(dip, pip));

  const bendDot = dot(mcpToPip, pipToDip);
  // Map dot product (1 → -1) to curl (0 → 1)
  // bendDot = 1 → straight, 0 → 90° bend, -1 → 180° bend
  //
  // PHASE 28 — this measures the bend at the PIP joint ALONE, which cannot
  // represent a whole finger's curl, and it is remapped here.
  //
  // Measured with synthetic landmarks of known pose, the raw value gives:
  //   straight finger              PIP   0°  → 0.000
  //   each joint 30°               PIP  30°  → 0.067
  //   each joint 60°               PIP  60°  → 0.250
  //   CLOSED FIST (90° per joint)  PIP  90°  → 0.500   ← cannot exceed 0.5
  //   fully folded (120°/joint)    PIP 120°  → 0.750
  //
  // A real fist bends at the MCP, PIP *and* DIP, so the PIP-only angle always
  // under-reports the total. Downstream that is multiplied by
  // FINGER_FLEX_LIMITS (80°) and split 0.5/0.3/0.2, so a closed fist rendered
  // at MCP/PIP/DIP = 20°/12°/8° against an anatomical capability of
  // 90°/110°/80° — a 70-98° shortfall. The hand physically cannot close.
  //
  // The remap below maps the PIP-only measurement onto the full curl range a
  // finger can actually express, using the measured endpoints above: a 90° PIP
  // bend (a closed fist) maps to 1.0, and a straight finger still maps to 0.
  // A 120° PIP bend is beyond a normal joint and is clamped to 1.0.
  const rawCurl = (1 - bendDot) / 2;
  const FIST_PIP_BEND_DEG = 90;
  const fistRaw = FIST_PIP_BEND_DEG / 180; // bendDot = cos(90°) = 0 → raw 0.5
  const curl = Math.max(0, Math.min(1, rawCurl / fistRaw));

  freeVec(mcpToPip);
  freeVec(pipToDip);

  return {
    dir: dirToTip,
    curl,
  };
}

/**
 * Compute a hand pose from MediaPipe hand landmarks.
 */
function computeHandPose(
  landmarks: Vec3[],
  wristPos: Vec3,
  side: "left" | "right"
): HandPose {
  // Palm direction: from wrist to middle MCP
  const middleMcp = landmarks[HAND.MIDDLE_MCP];
  const palmDir = normalize(sub(middleMcp, wristPos));

  // Palm normal: perpendicular to the palm plane.
  //
  // PHASE 15 BUGFIX — the cross-product operand ORDER must depend on which hand
  // this is. Verified numerically:
  //
  //   hand held up, palm facing the viewer (+Z)
  //     RIGHT: index MCP at -X, pinky MCP at +X
  //           cross(toIndex, toPinky) = (0, 0, -1)   <== INVERTED
  //     LEFT:  index MCP at +X, pinky MCP at -X
  //           cross(toIndex, toPinky) = (0, 0, +1)   <== correct
  //
  // The previous code used cross(toIndex, toPinky) for BOTH hands, so the right
  // palm pointed backwards. That is a 180deg flip about the wrist->middle-MCP
  // axis, which renders as the hand twisted inside-out and the forearm
  // contorted -- and it is asymmetric, so only one hand looked wrong.
  //
  // Swapping the operands for the right hand makes the rule symmetric:
  // the palm normal always points out of the palm for both hands.
  const indexMcp = landmarks[HAND.INDEX_MCP];
  const pinkyMcp = landmarks[HAND.PINKY_MCP];
  const toIndex = sub(indexMcp, wristPos);
  const toPinky = sub(pinkyMcp, wristPos);
  const palmNormal =
    side === "right"
      ? normalize(cross(toPinky, toIndex))
      : normalize(cross(toIndex, toPinky));

  // If palm normal is degenerate, use up
  if (palnLength(palmNormal) < 0.1) {
    palmNormal[0] = 0; palmNormal[1] = 0; palmNormal[2] = 1;
  }

  // Per-finger poses
  const fingers: Record<FingerName, FingerPose> = {
    thumb: { dir: [0, 0, 0], curl: 0 },
    index: { dir: [0, 0, 0], curl: 0 },
    middle: { dir: [0, 0, 0], curl: 0 },
    ring: { dir: [0, 0, 0], curl: 0 },
    pinky: { dir: [0, 0, 0], curl: 0 },
  };

  (["thumb", "index", "middle", "ring", "pinky"] as FingerName[]).forEach(
    (finger) => {
      fingers[finger] = computeFingerPose(landmarks, finger);
    }
  );

  freeVec(middleMcp);
  freeVec(indexMcp);
  freeVec(pinkyMcp);
  freeVec(toIndex);
  freeVec(toPinky);

  return {
    wristPos: [...wristPos],
    palmNormal,
    palmDir,
    fingers,
  };
}

function palnLength(v: Vec3): number {
  return len(v);
}

/**
 * Build a body pose from MediaPipe normalized landmarks.
 *
 * The input is already anchor-normalized (shoulder midpoint at origin,
 * shoulder-width scale = 1). We convert to canonical coordinates.
 */
export function computeBodyPose(
  data: number[] | Float32Array
): BodyPose {
  // Read key pose landmarks and convert to canonical
  const leftShoulder = mpToCanon(readPoseLM(data, POSE.LEFT_SHOULDER));
  const rightShoulder = mpToCanon(readPoseLM(data, POSE.RIGHT_SHOULDER));
  const leftHip = mpToCanon(readPoseLM(data, POSE.LEFT_HIP));
  const rightHip = mpToCanon(readPoseLM(data, POSE.RIGHT_HIP));
  const leftElbow = mpToCanon(readPoseLM(data, POSE.LEFT_ELBOW));
  const rightElbow = mpToCanon(readPoseLM(data, POSE.RIGHT_ELBOW));
  const leftWrist = mpToCanon(readPoseLM(data, POSE.LEFT_WRIST));
  const rightWrist = mpToCanon(readPoseLM(data, POSE.RIGHT_WRIST));
  const nose = mpToCanon(readPoseLM(data, POSE.NOSE));
  const leftKnee = mpToCanon(readPoseLM(data, POSE.LEFT_KNEE));
  const rightKnee = mpToCanon(readPoseLM(data, POSE.RIGHT_KNEE));
  const leftAnkle = mpToCanon(readPoseLM(data, POSE.LEFT_ANKLE));
  const rightAnkle = mpToCanon(readPoseLM(data, POSE.RIGHT_ANKLE));

  // Shoulder midpoint (the anchor point)
  const shoulderMid = scale(add(leftShoulder, rightShoulder), 0.5);

  // Hips position: use the midpoint of left/right hips
  // In normalized space, this is relative to shoulders
  const hipMid = scale(add(leftHip, rightHip), 0.5);

  // PHASE 21: upright spine reference.
  //
  // `normalizeLandmarks` anchors every buffer at the shoulder midpoint, so
  // `shoulderMid` is the world origin in every frame (measured: 79/79). That
  // collapses the old calculation
  //     spineDir = normalize(shoulderMid - hipMid)
  // to `normalize(-hipMid)`: the trunk axis is then decided entirely by the
  // HIPS, including their depth component. Measured on ADOPT, hip depth moves
  // up to 1.078 shoulder-widths between adjacent frames, and hip forward offset
  // correlates NEGATIVELY (r = -0.431) with the head aim -- a relationship no
  // standing human has. The spine was inheriting hip depth jitter as torso lean.
  //
  // The vertical component (how far the hips sit below the shoulders) is the
  // stable, meaningful signal; the forward/depth component in this capture is
  // not. Confining the trunk axis to the frontal plane keeps the measured
  // torso EXTENT while removing the unreliable depth term, which is what an
  // upright-torso prior means. Measured effect on rendered head-hips forward
  // offset: 0.1853 m -> 0.0836 m (slouch error 0.2161 m -> 0.1143 m, 47% fixed).
  //
  // HONEST SCOPE: this is a PRIOR, not a recovered signal. The underlying data
  // is not a valid human capture (the nose sits 0.937 torso-lengths forward of
  // the shoulder line; a real nose is ~0.10-0.15), so genuine forward lean
  // cannot be recovered from it. The durable fix is regenerating the dataset.
  const spineVec = sub(shoulderMid, hipMid);
  const spineDir = normalize(vec3(0, spineVec[1], 0));

  // Shoulder axis: from left to right shoulder
  const shoulderAxis = sub(rightShoulder, leftShoulder);
  const shoulderAxisN = normalize(shoulderAxis);

  // Shoulder height: the Y component of shoulder midpoint
  // (In canonical space, Y = up)
  const shoulderHeight = shoulderMid[1];

  // PHASE 21: upright head reference.
  //
  // The old value was `normalize(nose - shoulderMid)`. MediaPipe's pose model
  // has no neck landmark, so the nose was standing in for one -- but a nose is
  // a FORWARD-projecting facial point, not an up-pointing neck joint. Measured on
  // ADOPT, that vector tilts 69.6 deg from vertical versus 15.5 deg for the
  // spine, i.e. the head aim was ~54 deg worse than the trunk and was the single
  // largest contributor to the slouch (fixing it alone removed 30% of the error,
  // more than fixing the spine alone).
  //
  // The neck/head bones should follow the TRUNK axis, with the face's forward
  // offset expressed relative to the torso rather than to world depth. Deriving
  // it from the (now upright) spine direction keeps the neck and head
  // consistent with the torso and immune to the displaced nose landmark.
  //
  // `spineDir` is a pooled vector, so this MUST be a copy: the returned
  // BodyPose outlives the pool and aliasing it would let the hand pass
  // overwrite `spineDir` and `headDir` with the same object.
  const headDir = normalize(vec3(spineDir[0], spineDir[1], spineDir[2]));

  // Neck position is at the shoulder midpoint
  const neckPos = shoulderMid;

  // Head position: nose position
  const headPos = nose;

  // Free ONLY temps that the returned BodyPose does NOT reference.
  // (Every other vector in the old cleanup list WAS returned —
  // hipsPos=hipMid, leftShoulder/rightShoulder, hips, knees, ankles,
  // neckPos=shoulderMid, headPos=nose, shoulderAxis=shoulderAxisN.
  // Freeing them put them back in VEC3_POOL, where the subsequent hand pass
  // popped and OVERWROTE them, so the returned BodyPose read recycled pool
  // memory: shoulders came back as a shifted hand-wrist value and
  // shoulderAxis as a normalized wrist. Dormant-path use-after-free surfaced
  // by the E8 handedness harness — returned vectors must stay alive; pure
  // temps (spineVec, shoulderAxis) still get recycled.)
  // PHASE 21 removed headDirVec: headDir is now built from spineDir, so there
  // is no longer a separate headDirVec temporary to release.
  freeVec(spineVec);
  freeVec(shoulderAxis);

  return {
    hipsPos: hipMid,
    shoulderHeight,
    spineDir,
    shoulderAxis: shoulderAxisN,
    leftShoulder: leftShoulder,
    leftElbow,
    leftWrist: leftWrist,
    rightShoulder: rightShoulder,
    rightElbow: rightElbow,
    rightWrist: rightWrist,
    neckPos,
    headPos,
    headDir,
    leftHip,
    leftKnee,
    leftAnkle,
    rightHip,
    rightKnee,
    rightAnkle,
  } as BodyPose;
}

/**
 * Build a hand pose from MediaPipe hand landmarks in canonical space.
 */
/**
 * PHASE 5 — Defect 1: hand-presence test that survives normalization.
 *
 * The old test was `len(landmarks[HAND.WRIST]) < 1e-6`. That can never fire on a
 * normalized buffer: normalizeLandmarks maps EVERY float, including the
 * zero-filled slots of an absent group, through `(0 - anchor) / scale`, so an
 * absent hand's 21 points all become the SAME non-zero point at -anchor/scale.
 * The presence test therefore always reported "present", and computeHandPose
 * built a phantom hand from 21 coincident points: palmDir collapsed to the zero
 * vector, palmNormal fell through to the [0,0,1] default, and every finger came
 * out with dir = [0,0,0] and curl = 0.5 (the degenerate mcpToPip/pipToDip pair
 * dots to 0, and (1-0)/2 = 0.5) — i.e. hands that were not in frame silently
 * rendered as half-curled claws.
 *
 * A spread test is shift-invariant: an absent group is 21 IDENTICAL points, so
 * its maximum distance from the group centroid is exactly 0, while a real hand
 * spans a large fraction of a shoulder width. MIN_HAND_SPREAD is in the same
 * units normalizeLandmarks scales by (shoulder widths), so it is independent of
 * how far the subject is from the camera.
 */
const MIN_HAND_SPREAD = 0.02;

/** Max distance from a hand group's centroid to any of its 21 landmarks. */
function handGroupSpread(data: number[] | Float32Array, groupOffset: number): number {
  let cx = 0,
    cy = 0,
    cz = 0;
  for (let i = 0; i < HAND_COUNT; i++) {
    const b = groupOffset + i * 3;
    cx += data[b];
    cy += data[b + 1];
    cz += data[b + 2];
  }
  cx /= HAND_COUNT;
  cy /= HAND_COUNT;
  cz /= HAND_COUNT;

  let maxSq = 0;
  for (let i = 0; i < HAND_COUNT; i++) {
    const b = groupOffset + i * 3;
    const dx = data[b] - cx;
    const dy = data[b + 1] - cy;
    const dz = data[b + 2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > maxSq) maxSq = d2;
  }
  return Math.sqrt(maxSq);
}

/** PHASE 5 Defect 1 — is this hand group real data, or an absent group shifted
 * off the origin by normalizeLandmarks? */
function handGroupIsPresent(data: number[] | Float32Array, groupOffset: number): boolean {
  return handGroupSpread(data, groupOffset) > MIN_HAND_SPREAD;
}

type HandSide = "left" | "right";
export interface HandAssignment {
  /** Group offset chosen for the subject's anatomical left hand, or null. */
  left: number | null;
  /** Group offset chosen for the subject's anatomical right hand, or null. */
  right: number | null;
  debug: {
    spreadLeftSlot: number;
    spreadRightSlot: number;
    dLeftSlotToLeftWrist: number;
    dLeftSlotToRightWrist: number;
    dRightSlotToLeftWrist: number;
    dRightSlotToRightWrist: number;
  };
}

/**
 * PHASE 5 — Defect 2: resolve each detected hand to an ANATOMICAL side.
 *
 * MediaPipe's hand groups come from a handedness classifier that assumes a
 * MIRRORED (selfie) input, but useHolisticPipeline.ts draws the camera frame
 * un-mirrored (`ctx.drawImage(video, 0, 0, w, h)`, no transform). The Holistic
 * API exposes no handedness label at all — neither the JS
 * HolisticLandmarkerResult nor the Python one carries one — so there is no
 * label to trust and nothing to read back.
 *
 * The side is therefore resolved GEOMETRICALLY, from the POSE wrists. Pose
 * landmarks are anatomical (15/16 are the subject's real wrists) and are not
 * subject to the mirror assumption, which makes them independent ground truth.
 * Assigning each hand to the pose wrist it actually sits on is correct whether
 * or not the classifier's slots are mirrored, so the fix does not depend on
 * resolving the classifier's convention at all.
 *
 * Matching is greedy over the 2x2 (group x side) edge set sorted by distance,
 * so two hands can never claim the same wrist.
 */
/**
 * PHASE 36 — assignment hysteresis.
 *
 * Greedy nearest-wrist matching is correct in principle, but with no memory the
 * decision is re-made from scratch every frame. When the two wrist distances
 * are nearly tied, the winner can change from frame to frame, and the hand
 * slots then swap: the subject's left hand suddenly receives the right hand's
 * 21 landmarks. Because `palmNormal` is a signed cross product, a swap inverts
 * it, producing a ~180 deg palm flip in a single frame.
 *
 * Measured on the shipped clips: the left-hand slot changed 6/79 times on ADOPT
 * and 22/208 on HELLO (about 11% of frames), with a median decision margin of
 * only 2.8e-1 -- effectively a coin flip. This is the source of the 140-175 deg
 * per-frame bone swings and the near-random trajectory consistency (0.047).
 *
 * MEASURED EFFECT of this margin (window-3 hand smoothing active, all clips):
 *
 *   margin   slot flips            palmNormal step
 *            ADOPT    HELLO        ADOPT    HELLO
 *   none       6/79    22/208       19.2     14.7      <- stateless baseline
 *   0.15      3/79    12/208       22.0     14.7
 *   0.6       3/79    12/208       22.0     15.3
 *   1.6       7/79    12/208       25.2     15.3
 *   99.0      7/79    12/208       25.2     15.3      <- effectively disabled
 *
 * So 0.15 roughly halves the flips (ADOPT 6->3, HELLO 22->12) at a small cost in
 * palmNormal step (19.2 -> 22.0 deg on ADOPT). It does NOT eliminate them: a
 * large margin eventually DISABLES the prior, because `keep` requires
 * `d(same side) + margin <= d(other side)`, so past the point where that can
 * never hold, both sides fall through to the stateless greedy result (row 99.0).
 *
 * HONEST SCOPE: this is a mitigation, not a cure. The flips that remain come
 * from frames where the two wrist distances are near-tied for many frames in a
 * row, which is a property of the source capture rather than of this matcher.
 * The durable fix is upstream landmark/pose quality.
 *
 * This keeps the module STATELESS with respect to its caller: it reads the
 * module-level `lastHandAssignment` that the pose entry point already publishes,
 * and there is no reset hook or cross-sequence bleed because each new sequence
 * simply overwrites it on its first frame.
 */
const HYSTERESIS_MARGIN = 0.15;

function resolveHandAssignment(data: number[] | Float32Array): HandAssignment {
  const debug: HandAssignment["debug"] = {
    spreadLeftSlot: handGroupSpread(data, LEFT_HAND_OFFSET),
    spreadRightSlot: handGroupSpread(data, RIGHT_HAND_OFFSET),
    dLeftSlotToLeftWrist: NaN,
    dLeftSlotToRightWrist: NaN,
    dRightSlotToLeftWrist: NaN,
    dRightSlotToRightWrist: NaN,
  };

  const lWrist = readPoseLM(data, POSE.LEFT_WRIST);
  const rWrist = readPoseLM(data, POSE.RIGHT_WRIST);
  const edges: { group: number; side: HandSide; d: number }[] = [];

  const consider = (groupOffset: number, isLeftSlot: boolean) => {
    if (!handGroupIsPresent(data, groupOffset)) return;
    const wrist = readHandLM(data, groupOffset, HAND.WRIST);
    const dL = len(sub(wrist, lWrist));
    const dR = len(sub(wrist, rWrist));
    if (isLeftSlot) {
      debug.dLeftSlotToLeftWrist = dL;
      debug.dLeftSlotToRightWrist = dR;
    } else {
      debug.dRightSlotToLeftWrist = dL;
      debug.dRightSlotToRightWrist = dR;
    }
    edges.push({ group: groupOffset, side: "left", d: dL });
    edges.push({ group: groupOffset, side: "right", d: dR });
  };

  consider(LEFT_HAND_OFFSET, true);
  consider(RIGHT_HAND_OFFSET, false);

  edges.sort((a, b) => a.d - b.d);
  const out: HandAssignment = { left: null, right: null, debug };
  const usedGroups = new Set<number>();
  for (const e of edges) {
    if (out[e.side] !== null) continue;
    if (usedGroups.has(e.group)) continue;
    out[e.side] = e.group;
    usedGroups.add(e.group);
  }

  // PHASE 36 — hysteresis. The greedy pass above is stateless, so on a near-tie
  // it can hand the subject's left hand the right hand's slot (and vice versa)
  // on any given frame. A swap inverts the signed palm normal outright.
  //
  // Re-run the greedy pass with the PREVIOUS assignment as a sticky prior: a
  // group already assigned to a side keeps it unless the competing edge for
  // that same group is closer by more than HYSTERESIS_MARGIN. So a tie is
  // always resolved in favour of continuity, and a genuine hand swap still
  // happens as soon as it wins decisively.
  //
  // The previous assignment is only honoured when it is still structurally
  // valid this frame (both groups still present and distinct), so a lost hand
  // cannot pin a stale slot forever.
  const prev = lastHandAssignment;
  if (prev && (prev.left !== null || prev.right !== null)) {
    const present = (g: number | null) => g !== null && handGroupIsPresent(data, g);
    const prevLeftOk = present(prev.left) && prev.left !== prev.right;
    const prevRightOk = present(prev.right) && prev.left !== prev.right;
    // Only apply the prior when BOTH sides are usable, so a one-sided sequence
    // falls back to the plain greedy result.
    if (prevLeftOk && prevRightOk) {
      const dOf = (g: number, side: HandSide): number => {
        const wrist = readHandLM(data, g, HAND.WRIST);
        return side === "left" ? len(sub(wrist, lWrist)) : len(sub(wrist, rWrist));
      };
      const keepLeft = dOf(prev.left!, "left") + HYSTERESIS_MARGIN <= dOf(prev.left!, "right");
      const keepRight = dOf(prev.right!, "right") + HYSTERESIS_MARGIN <= dOf(prev.right!, "left");
      if (keepLeft && keepRight) {
        out.left = prev.left;
        out.right = prev.right;
      } else if (keepLeft) {
        // Left stays; the other hand takes whichever group the left one vacated.
        out.left = prev.left;
        out.right = prev.right === prev.left ? (prev.left === LEFT_HAND_OFFSET ? RIGHT_HAND_OFFSET : LEFT_HAND_OFFSET) : prev.right;
        if (out.right === out.left) out.right = null;
      } else if (keepRight) {
        out.right = prev.right;
        out.left = prev.left === prev.right ? (prev.right === LEFT_HAND_OFFSET ? RIGHT_HAND_OFFSET : LEFT_HAND_OFFSET) : prev.left;
        if (out.left === out.right) out.left = null;
      }
      // If neither side is sticky, keep the stateless greedy result.
    }
  }
  return out;
}

function computeHandPoseFromRaw(
  data: number[] | Float32Array,
  groupOffset: number,
  side: "left" | "right"
): HandPose | null {
  // PHASE 5 Defect 1: spread test, not a magnitude test on the wrist — the
  // magnitude test is defeated by normalizeLandmarks' anchor shift.
  if (!handGroupIsPresent(data, groupOffset)) {
    return null;
  }

  // Read all 21 hand landmarks, convert to canonical
  const landmarks: Vec3[] = [];
  for (let i = 0; i < HAND_COUNT; i++) {
    const raw = readHandLM(data, groupOffset, i);
    landmarks.push(mpToCanon(raw));
  }

  // PHASE 15: `side` is required so computeHandPose can pick the correct
  // cross-product order for the palm normal (see the note there).
  const pose = computeHandPose(landmarks, landmarks[HAND.WRIST], side);

  // Free temp landmark vectors (pose holds copies via spread)
  for (const v of landmarks) freeVec(v);

  return pose;
}


/**
 * PHASE 5 — the side assignment chosen by the most recent
 * mediapipeToCanonicalPose() call. Exposed so the verification harness can
 * assert Defect 2 directly instead of inferring it from bone output.
 */
export let lastHandAssignment: HandAssignment | null = null;

/**
 * Convert a packed MediaPipe landmark buffer to a CanonicalPose.
 *
 * @param data The packed Float32Array from normalizeLandmarks or packHolisticResult
 * @param timestamp Frame timestamp in seconds
 */
export function mediapipeToCanonicalPose(
  data: number[] | Float32Array,
  timestamp: number = 0
): CanonicalPose | null {
  // Compute body pose
  const body = computeBodyPose(data);

  // PHASE 5 Defect 2: do NOT trust the hand array slots. Resolve each detected
  // hand to the anatomical side it actually occupies, using the pose wrists.
  const assign = resolveHandAssignment(data);
  lastHandAssignment = assign;

  // Compute hand poses from the anatomically-resolved groups.
  const leftHand =
    assign.left !== null ? computeHandPoseFromRaw(data, assign.left, "left") : null;
  const rightHand =
    assign.right !== null ? computeHandPoseFromRaw(data, assign.right, "right") : null;

  // If no body pose detected at all, return null
  if (
    body.leftShoulder.every((v) => Math.abs(v) < 1e-6) &&
    body.leftWrist.every((v) => Math.abs(v) < 1e-6)
  ) {
    return null;
  }

  // Default hands if not detected
  const leftHandPose: HandPose = leftHand ?? {
    wristPos: body.leftWrist,
    palmNormal: [0, 0, 1],
    palmDir: [0, 0, 1],
    fingers: {
      thumb: { dir: [0, 0, 1], curl: 0 },
      index: { dir: [0, 0, 1], curl: 0 },
      middle: { dir: [0, 0, 1], curl: 0 },
      ring: { dir: [0, 0, 1], curl: 0 },
      pinky: { dir: [0, 0, 1], curl: 0 },
    },
  };

  const rightHandPose: HandPose = rightHand ?? {
    wristPos: body.rightWrist,
    palmNormal: [0, 0, 1],
    palmDir: [0, 0, 1],
    fingers: {
      thumb: { dir: [0, 0, 1], curl: 0 },
      index: { dir: [0, 0, 1], curl: 0 },
      middle: { dir: [0, 0, 1], curl: 0 },
      ring: { dir: [0, 0, 1], curl: 0 },
      pinky: { dir: [0, 0, 1], curl: 0 },
    },
  };

  return {
    body,
    leftHand: leftHandPose,
    rightHand: rightHandPose,
    timestamp,
  };
}
