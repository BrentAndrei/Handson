/**
 * Gate for the video → avatar-data capture engine (src/lib/videoCapture.ts).
 *
 * The engine's whole value proposition is that a clip extracted from a
 * tutorial video is indistinguishable from the shipped corpus: same
 * 345-float frame layout, same shoulder-midpoint normalization, same
 * 40-point lip subset. These checks assert that contract directly,
 * without a browser, by driving the exact pack → normalize → select
 * chain the extraction loop uses, against a synthetic Holistic result
 * with known geometry:
 *
 *   V1  one frame packs to exactly FEATURE_DIM (345) floats
 *   V2  the shoulder midpoint lands at the origin (anchor = 0,0,0)
 *   V3  shoulder width scales to exactly 1
 *   V4  the pose block (0..99) carries the normalized pose landmarks
 *   V5  the left-hand block (99..162) carries the normalized left hand
 *   V6  the right-hand block (162..225) carries the normalized right hand
 *   V7  the lip block (225..345) carries the 40 LIP_LANDMARK_INDICES
 *       of the normalized face landmarks, in order
 *   V8  an absent landmark group zero-fills its block (pack flags false)
 *   V9  buildAvatarDataJson round-trips: label/frameCount/frames,
 *       frames.length === frameCount * 345
 *   V10 avatarDataFilename sanitizes non-alphanumerics to "_"
 *   V11 the shipped corpus obeys the same frame-count contract
 *       (KNOW.json: frameCount * 345 === frames.length)
 *   V12 mirrorFrame is an involution: mirror(mirror(x)) === x
 *   V13 mirrorFrame negates x, swaps the symmetric pose pairs,
 *       exchanges the two hand blocks, and leaves the lip block
 *       values in place (x negated, no index swap)
 *   V14 testClipAgainstModel: window count, probability
 *       aggregation, agree flag, and the too-short-clip error
 *       (tiny deterministic tfjs model — zero kernel, bias
 *       [1, 0] — so class 0 always wins)
 */
import * as fs from "node:fs";
import * as path from "node:path";
import {
  packHolisticResult,
} from "../src/mediapipe/pack";
import { normalizeLandmarks } from "../src/utils/normalizeLandmarks";
import {
  selectFeatures,
  FEATURE_DIM,
  LIP_LANDMARK_INDICES,
} from "../src/features/featureSelect";
import {
  POSE_COUNT,
  HAND_COUNT,
  FACE_COUNT,
  POSE_OFFSET,
  LEFT_HAND_OFFSET,
  RIGHT_HAND_OFFSET,
} from "../src/mediapipe/types";
import {
  buildAvatarDataJson,
  avatarDataFilename,
  mirrorFrame,
} from "../src/lib/videoCapture";
import { testClipAgainstModel } from "../src/lib/clipTester";
import {
  DEFAULT_STRIDE,
  WINDOW_SIZE,
} from "../src/features/windowBuffer";
import * as tf from "@tensorflow/tfjs";
import type { NormalizedLandmark, HolisticLandmarkerResult } from "@mediapipe/tasks-vision";

let checks = 0;
let failures = 0;

function check(name: string, ok: boolean, detail = ""): void {
  checks++;
  if (!ok) {
    failures++;
    console.log(`  FAIL ${name}${detail ? `: ${detail}` : ""}`);
  } else {
    console.log(`  ok   ${name}`);
  }
}

const pt = (x: number, y: number, z: number): NormalizedLandmark => ({
  x,
  y,
  z,
  visibility: 1,
});

/** A synthetic, geometrically known Holistic result.
 *  Shoulders at (0.42, 0.42, 0) and (0.58, 0.42, 0):
 *  anchor = (0.5, 0.42, 0), shoulder width = 0.16. */
function syntheticResult() {
  const pose: NormalizedLandmark[] = [];
  for (let i = 0; i < POSE_COUNT; i++) {
    pose.push(pt(0.3 + i * 0.011, 0.2 + i * 0.017, i * 0.003));
  }
  pose[0] = pt(0.5, 0.35, 0.01); // nose
  pose[11] = pt(0.42, 0.42, 0); // left shoulder
  pose[12] = pt(0.58, 0.42, 0); // right shoulder
  pose[23] = pt(0.44, 0.62, 0); // left hip
  pose[24] = pt(0.56, 0.62, 0); // right hip

  const leftHand: NormalizedLandmark[] = [];
  for (let i = 0; i < HAND_COUNT; i++) {
    leftHand.push(pt(0.1 + i * 0.007, 0.3 + i * 0.013, i * 0.002));
  }
  const rightHand: NormalizedLandmark[] = [];
  for (let i = 0; i < HAND_COUNT; i++) {
    rightHand.push(pt(0.7 + i * 0.006, 0.5 + i * 0.011, i * 0.004));
  }
  const face: NormalizedLandmark[] = [];
  for (let i = 0; i < FACE_COUNT; i++) {
    face.push(pt(0.2 + i * 0.0009, 0.3 + i * 0.0011, i * 0.0005));
  }

  return {
    poseLandmarks: [pose],
    leftHandLandmarks: [leftHand],
    rightHandLandmarks: [rightHand],
    faceLandmarks: [face],
  };
}

const ANCHOR = [0.5, 0.42, 0] as const;
const SCALE = 0.16;
const norm = (p: NormalizedLandmark): [number, number, number] => [
  (p.x - ANCHOR[0]) / SCALE,
  (p.y - ANCHOR[1]) / SCALE,
  (p.z - ANCHOR[2]) / SCALE,
];

const result = syntheticResult();
const packed = packHolisticResult(
  result as unknown as HolisticLandmarkerResult,
  0
);
const normalized = normalizeLandmarks(packed);
const features = selectFeatures(normalized.vector);

// V1
check(
  "V1 one frame packs to FEATURE_DIM floats",
  features.length === FEATURE_DIM && FEATURE_DIM === 345,
  `length ${features.length}`
);

// V2 — shoulders at the origin
const lShoulder = [features[POSE_OFFSET + 11 * 3], features[POSE_OFFSET + 11 * 3 + 1], features[POSE_OFFSET + 11 * 3 + 2]];
const rShoulder = [features[POSE_OFFSET + 12 * 3], features[POSE_OFFSET + 12 * 3 + 1], features[POSE_OFFSET + 12 * 3 + 2]];
check(
  "V2 shoulder midpoint at origin",
  Math.abs((lShoulder[0] + rShoulder[0]) / 2) < 1e-6 &&
    Math.abs((lShoulder[1] + rShoulder[1]) / 2) < 1e-6 &&
    Math.abs((lShoulder[2] + rShoulder[2]) / 2) < 1e-6,
  `midpoint ${(lShoulder[0] + rShoulder[0]) / 2}`
);

// V3 — shoulder width is 1
const shoulderWidth = Math.hypot(
  rShoulder[0] - lShoulder[0],
  rShoulder[1] - lShoulder[1],
  rShoulder[2] - lShoulder[2]
);
check("V3 shoulder width scales to 1", Math.abs(shoulderWidth - 1) < 1e-6, String(shoulderWidth));

// V4/V5/V6 — blocks carry the normalized landmarks
let poseOk = true;
let leftOk = true;
let rightOk = true;
const poseLms = result.poseLandmarks[0];
const leftLms = result.leftHandLandmarks[0];
const rightLms = result.rightHandLandmarks[0];
for (let i = 0; i < POSE_COUNT; i++) {
  const n = norm(poseLms[i]);
  const b = POSE_OFFSET + i * 3;
  if (Math.abs(features[b] - n[0]) > 1e-6 || Math.abs(features[b + 1] - n[1]) > 1e-6 || Math.abs(features[b + 2] - n[2]) > 1e-6) {
    poseOk = false;
  }
}
for (let i = 0; i < HAND_COUNT; i++) {
  const n = norm(leftLms[i]);
  const b = LEFT_HAND_OFFSET + i * 3;
  if (Math.abs(features[b] - n[0]) > 1e-6 || Math.abs(features[b + 1] - n[1]) > 1e-6 || Math.abs(features[b + 2] - n[2]) > 1e-6) {
    leftOk = false;
  }
  const nr = norm(rightLms[i]);
  const br = RIGHT_HAND_OFFSET + i * 3;
  if (Math.abs(features[br] - nr[0]) > 1e-6 || Math.abs(features[br + 1] - nr[1]) > 1e-6 || Math.abs(features[br + 2] - nr[2]) > 1e-6) {
    rightOk = false;
  }
}
check("V4 pose block carries normalized landmarks", poseOk);
check("V5 left-hand block carries normalized landmarks", leftOk);
check("V6 right-hand block carries normalized landmarks", rightOk);

// V7 — lip block. In the SELECTED layout it starts at
// pose(99) + leftHand(63) + rightHand(63) = 225.
const faceLms = result.faceLandmarks[0];
let lipOk = true;
for (let k = 0; k < LIP_LANDMARK_INDICES.length; k++) {
  const idx = LIP_LANDMARK_INDICES[k];
  const n = norm(faceLms[idx]);
  const b = 225 + k * 3;
  if (Math.abs(features[b] - n[0]) > 1e-6 || Math.abs(features[b + 1] - n[1]) > 1e-6 || Math.abs(features[b + 2] - n[2]) > 1e-6) {
    lipOk = false;
  }
}
check(
  "V7 lip block carries the 40 lip landmarks in order",
  lipOk && LIP_LANDMARK_INDICES.length === 40 && features.length === 345,
  `lip count ${LIP_LANDMARK_INDICES.length}`
);

// V8 — absent groups zero-fill
const poseOnly = {
  poseLandmarks: [result.poseLandmarks[0]],
  leftHandLandmarks: [],
  rightHandLandmarks: [],
  faceLandmarks: [],
};
const packed2 = packHolisticResult(
  poseOnly as unknown as HolisticLandmarkerResult,
  0
);
check(
  "V8 absent groups zero-fill and flag false",
  packed2.hasPose === true &&
    packed2.hasLeftHand === false &&
    packed2.hasRightHand === false &&
    packed2.hasFace === false,
  `flags ${packed2.hasPose}/${packed2.hasLeftHand}/${packed2.hasRightHand}/${packed2.hasFace}`
);

// V9 — JSON round-trip (3 identical frames)
const threeFrames: number[] = [];
for (let i = 0; i < 3; i++) threeFrames.push(...Array.from(features));
const json = buildAvatarDataJson({
  label: "TEST SIGN",
  frameCount: 3,
  frames: threeFrames,
  coverage: { frames: 3, pose: 3, leftHand: 3, rightHand: 3, face: 3 },
  fps: 30,
});
const parsed = JSON.parse(json);
check(
  "V9 avatar-data JSON round-trips",
  parsed.label === "TEST SIGN" &&
    parsed.frameCount === 3 &&
    Array.isArray(parsed.frames) &&
    parsed.frames.length === 3 * FEATURE_DIM,
  `frames ${parsed.frames?.length}`
);

// V10 — filename sanitization
check(
  "V10 filename sanitizes punctuation",
  avatarDataFilename("ALRIGHT, NO PROBLEM") === "ALRIGHT__NO_PROBLEM.json" &&
    avatarDataFilename("KNOW") === "KNOW.json",
  avatarDataFilename("ALRIGHT, NO PROBLEM")
);

// V11 — the shipped corpus obeys the same contract
const corpusPath = path.join(process.cwd(), "public", "avatar-data", "KNOW.json");
const corpusRaw = fs.readFileSync(corpusPath, "utf8");
const corpus = JSON.parse(corpusRaw);
check(
  "V11 shipped corpus matches the frame contract",
  corpus.label === "KNOW" &&
    typeof corpus.frameCount === "number" &&
    Array.isArray(corpus.frames) &&
    corpus.frames.length === corpus.frameCount * FEATURE_DIM,
  `frameCount ${corpus.frameCount}, floats ${corpus.frames?.length}`
);

// V12 — mirrorFrame is an involution
const once = mirrorFrame(features);
const twice = mirrorFrame(once);
let involutionOk = true;
for (let i = 0; i < FEATURE_DIM; i++) {
  if (Math.abs(twice[i] - features[i]) > 1e-6) {
    involutionOk = false;
    break;
  }
}
check("V12 mirrorFrame is an involution", involutionOk);

// V13 — mirror geometry:
//   every x negated; symmetric pose pairs swapped (left eye 1
//   takes the negated right eye 4, shoulders 11/12 swap);
//   the two hand blocks exchange wholesale; the lip block keeps
//   its indices (only x is negated — no consumer reads it).
const mirrored = once;
const rEye = [features[4 * 3], features[4 * 3 + 1], features[4 * 3 + 2]];
const mLEye = [mirrored[1 * 3], mirrored[1 * 3 + 1], mirrored[1 * 3 + 2]];
const xNegated = Math.abs(mirrored[0] + features[0]) < 1e-6 &&
  Math.abs(mirrored[1] - features[1]) < 1e-6 &&
  Math.abs(mirrored[2] - features[2]) < 1e-6;
const eyeSwap =
  Math.abs(mLEye[0] + rEye[0]) < 1e-6 &&
  Math.abs(mLEye[1] - rEye[1]) < 1e-6 &&
  Math.abs(mLEye[2] - rEye[2]) < 1e-6;
const shoulderSwap =
  Math.abs(mirrored[11 * 3] + features[12 * 3]) < 1e-6 &&
  Math.abs(mirrored[12 * 3] + features[11 * 3]) < 1e-6;
let handSwapOk = true;
// The blocks exchange AFTER x is negated, so the mirrored
// left hand is the x-NEGATED right hand (and vice versa).
for (let i = 0; i < HAND_COUNT; i++) {
  const l = LEFT_HAND_OFFSET + i * 3;
  const r = RIGHT_HAND_OFFSET + i * 3;
  if (
    Math.abs(mirrored[l] + features[r]) > 1e-6 ||
    Math.abs(mirrored[l + 1] - features[r + 1]) > 1e-6 ||
    Math.abs(mirrored[l + 2] - features[r + 2]) > 1e-6 ||
    Math.abs(mirrored[r] + features[l]) > 1e-6 ||
    Math.abs(mirrored[r + 1] - features[l + 1]) > 1e-6 ||
    Math.abs(mirrored[r + 2] - features[l + 2]) > 1e-6
  ) {
    handSwapOk = false;
    break;
  }
}
let lipUntouchedOk = true;
for (let k = 0; k < LIP_LANDMARK_INDICES.length; k++) {
  const b = 225 + k * 3;
  if (
    Math.abs(mirrored[b] + features[b]) > 1e-6 ||
    Math.abs(mirrored[b + 1] - features[b + 1]) > 1e-6 ||
    Math.abs(mirrored[b + 2] - features[b + 2]) > 1e-6
  ) {
    lipUntouchedOk = false;
    break;
  }
}
check(
  "V13 mirror negates x, swaps pose pairs + hand blocks, lip in place",
  xNegated && eyeSwap && shoulderSwap && handSwapOk && lipUntouchedOk,
  `x=${xNegated} eye=${eyeSwap} shoulder=${shoulderSwap} hand=${handSwapOk} lip=${lipUntouchedOk}`
);

// V14 — the clip tester slices, predicts and aggregates correctly.
// A deterministic model: zero kernel means the output is the bias
// vector [1, 0] for EVERY window, so the mean is [1, 0] too and
// class 0 must win with confidence exactly 1.
const TEST_LABELS = ["TEST SIGN A", "TEST SIGN B"];
const testModel = tf.sequential({
  layers: [
    tf.layers.flatten({ inputShape: [WINDOW_SIZE, FEATURE_DIM] }),
    tf.layers.dense({ units: 2 }),
  ],
});
testModel.setWeights([
  tf.zeros([WINDOW_SIZE * FEATURE_DIM, 2]),
  tf.tensor1d([1, 0]),
]);

const testFrames: number[] = [];
for (let i = 0; i < 30; i++) {
  const frame = new Float32Array(FEATURE_DIM);
  frame[0] = i;
  testFrames.push(...Array.from(frame));
}
const expectedWindows = Math.floor((30 - WINDOW_SIZE) / DEFAULT_STRIDE) + 1;

const agreeResult = await testClipAgainstModel(
  testModel,
  testFrames,
  "TEST SIGN A",
  TEST_LABELS
);
check(
  "V14a clip test: window count + class 0 wins at 1.0",
  agreeResult.windows === expectedWindows &&
    agreeResult.top.length === 2 &&
    agreeResult.agree === true &&
    Math.abs(agreeResult.top[0].confidence - 1) < 1e-6 &&
    Math.abs(agreeResult.top[1].confidence - 0) < 1e-6,
  `windows ${agreeResult.windows}/${expectedWindows}, top ${agreeResult.top.map((p) => `${p.label}:${p.confidence}`).join(", ")}, agree ${agreeResult.agree}`
);

const disagreeResult = await testClipAgainstModel(
  testModel,
  testFrames,
  "TEST SIGN B",
  TEST_LABELS
);
check(
  "V14b clip test: disagree flag when the label is class 1",
  disagreeResult.agree === false &&
    disagreeResult.top[0].label === "TEST SIGN A",
  `agree ${disagreeResult.agree}, top ${disagreeResult.top[0]?.label}`
);

let shortClipThrew = false;
try {
  const shortFrames: number[] = [];
  for (let i = 0; i < WINDOW_SIZE - 1; i++) {
    shortFrames.push(...new Array(FEATURE_DIM).fill(0));
  }
  await testClipAgainstModel(
    testModel,
    shortFrames,
    "TEST SIGN A",
    TEST_LABELS
  );
} catch {
  shortClipThrew = true;
}
check("V14c clip test: too-short clip throws", shortClipThrew);

testModel.dispose();

console.log(
  failures === 0
    ? `\nVIDEO CAPTURE VERIFY PASSED (${checks} checks)`
    : `\nVIDEO CAPTURE VERIFY FAILED: ${failures} of ${checks} checks`
);
process.exit(failures === 0 ? 0 : 1);
