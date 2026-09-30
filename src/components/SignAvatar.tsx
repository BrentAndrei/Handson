import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { AvatarFrame, ParentWorldQuats } from "../avatar/retarget";
import { loadLabelData, loadAvatarData, hasLabelData, resetAvatarData, loadLabelDataCanonical, loadLabelDataMediaPipe } from "../avatar/dataLoader";
import { resolveSkeleton } from "../avatar/boneResolver";
import type { ResolvedSkeleton } from "../avatar/boneResolver";
import { retarget, applyBoneRotations } from "../avatar/retargeter";
import type { RetargetedPose } from "../avatar/retargeter";
import type { CanonicalPose } from "../avatar/canonicalPose";
import { RootTranslationSolver } from "../avatar/rootTranslation";
import { solveFooting } from "../avatar/footingSolver";

import { mediapipeToCanonicalPose } from "../avatar/mediapipeAdapter";

/** PHASE 9: Xbot's bind-pose floor. Measured from the real GLB — its lowest
 *  bone (mixamorigLeftToe_End) sits at y = -0.0029 m at bind, so the floor is
 *  that value, NOT y = 0. */
const FOOTING_FLOOR_Y = -0.0029;

interface SignAvatarProps {
  signSequence?: string[];
  /**
   * PHASE 10 — live webcam drive.
   *
   * When this ref holds a frame, the avatar is driven by the live MediaPipe
   * Holistic feed instead of a recorded clip. The value is a
   * `{ vector, seq }` pair: `seq` increments per inference result so the render
   * loop can tell a NEW frame from a re-render of the same one and retarget at
   * most once per inference.
   *
   * The frame goes through the exact same verified path as recorded data
   * (`mediapipeToCanonicalPose` -> `retarget`), so the live pose is subject to
   * the identical body/hand mapping, the identical root solver and the
   * identical footing solver. Nothing about the retargeting maths changes
   * between the two sources.
   */
  liveFrameRef?: React.MutableRefObject<{ vector: Float32Array; seq: number } | null>;
  /** True while the live source should be preferred over recorded playback. */
  liveActive?: boolean;
  playbackSpeed?: number;
  isPlaying?: boolean;
  currentFrameIndex?: number;
  onStepForward?: () => void;
  onStepBack?: () => void;
  onFrameChange?: (index: number, total: number) => void;
  onSignChange?: (label: string) => void;
  onModelReady?: (ready: boolean) => void;
}

const BONE_MAP: Record<string, string> = {
  Hips: "root",
  Spine: "torso",
  Head: "head",
  Neck: "neck",
  LeftShoulder: "leftShoulder",
  LeftArm: "leftUpperArm",
  LeftForeArm: "leftForearm",
  LeftHand: "leftHand",
  RightShoulder: "rightShoulder",
  RightArm: "rightUpperArm",
  RightForeArm: "rightForearm",
  RightHand: "rightHand",
};

const AVATAR_KEYS: (keyof AvatarFrame)[] = [
  "root", "hips", "torso", "head", "neck",
  "leftShoulder", "leftUpperArm", "leftForearm", "leftHand",
  "rightShoulder", "rightUpperArm", "rightForearm", "rightHand",
];

const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 2];

const QUAT_BONE_KEYS = new Set([
  "torso",
  "leftUpperArm", "leftForearm", "leftHand",
  "rightUpperArm", "rightForearm", "rightHand",
  "leftShoulder", "rightShoulder", "head", "neck",
]);
const SMOOTHING = 0.25;
const MAX_BONE_ANGLE = Math.PI * 0.9;  // 162 degrees - allows full signing arm movements

/**
 * PHASE 10 — reusable scratch for the render loop.
 *
 * Module-scoped rather than a `useRef` so it is allocated once per module load
 * instead of once per component instance, and so the animation loop closure
 * (installed once, lives for the component's lifetime) does not depend on
 * component state. Safe to share: the loop is single-threaded and each buffer
 * is fully consumed before the next line needs it.
 */
const boneNameScratch = new Set<string>();

/**
 * Root translation handling (PHASE 7).
 *
 * `RetargetedPose.hipsWorldPos` is the source pose's pelvis position, and it
 * lives in SHOULDER-WIDTH units around a shoulder-anchored origin -- not metres
 * around a world origin, despite the canonical doc comment. Measured on the real
 * fixtures: the shoulder midpoint is EXACTLY [0,0,0] in every normalized buffer
 * (normalizeLandmarks subtracts it), so absolute position is unrecoverable, and
 * the raw pelvis value is noisy enough that copying it straight onto
 * `model.position` threw the avatar ~1.6 units below the frame and made it drift
 * frame-to-frame. That is why this stayed locked through Phases 1-6.
 *
 * It is now re-enabled, but only as a DELTA through RootTranslationSolver:
 *   offset = (hips_now - hips_baseline) * metresPerShoulderWidth
 * with the baseline captured from the first frame of a sequence, a per-axis
 * human movement envelope, a per-frame slew limit, and a floor clamp derived
 * from the avatar's own bind-pose foot height. See avatar/rootTranslation.ts for
 * the measurements behind each limit.
 */
const APPLY_ROOT_TRANSLATION = true;

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export default function SignAvatar({
  signSequence,
  liveFrameRef,
  liveActive = false,
  playbackSpeed: externalSpeed = 0.25,
  isPlaying: externalPlaying = true,
  currentFrameIndex: externalFrameIndex,
  onStepForward,
  onStepBack,
  onFrameChange,
  onSignChange,
  onModelReady,
}: SignAvatarProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const signIndexRef = useRef(0);
  const lastSignTimeRef = useRef(Date.now());
  const allAvailableRef = useRef<string[]>([]);
  const animationIdRef = useRef(0);
  const timeRef = useRef(0);
  const allBoneNamesRef = useRef<string[]>([]);
  const avatarRef = useRef<{
    model: THREE.Group;
    bones: Record<string, THREE.Bone>;
  } | null>(null);
  const currentFramesRef = useRef<AvatarFrame[]>([]);
  const resolvedSkeletonRef = useRef<ResolvedSkeleton | null>(null);
  const currentRetargetedRef = useRef<RetargetedPose[]>([]);
  const transitionProgressRef = useRef<number>(0);
  const prevFrameRef = useRef<AvatarFrame | null>(null);
  const targetFrameRef2 = useRef<AvatarFrame | null>(null);
  const prevRetargetedRef = useRef<RetargetedPose | null>(null);
  const targetRetargetedRef = useRef<RetargetedPose | null>(null);
  const usedCanonicalRef = useRef(false);
  const transitionStartTimeRef = useRef(0);
  const transitionDuration = 0.4;
  const isAnimatingRef = useRef(false);
  const breathPhaseRef = useRef(0);
  const idleWeightRef = useRef(0);
  const modelReadyRef = useRef(false);
  const tmpQuat1 = useRef(new THREE.Quaternion());
  const tmpQuat2 = useRef(new THREE.Quaternion());
  const tmpEuler1 = useRef(new THREE.Euler());
  const bindPoseQuats = useRef<Record<string, THREE.Quaternion>>({});
  const parentWorldQuatsRef = useRef<Record<string, THREE.Quaternion>>({});
  const resultQuat = useRef(new THREE.Quaternion());
   const idleAxis = useRef(new THREE.Vector3(1, 0, 0));
  const tmpVec1 = useRef(new THREE.Vector3());
  const tmpVec2 = useRef(new THREE.Vector3());
  // PHASE 7: delta-based root translation (see avatar/rootTranslation.ts).
  const rootSolverRef = useRef(new RootTranslationSolver());
  const tmpRootOffset = useRef(new THREE.Vector3());
  // PHASE 10 — live-source bookkeeping. `liveSeqRef` records the last live frame
  // consumed so a re-render with the same frame does not retarget twice;
  // `livePrevPoseRef` holds the previously applied live pose so consecutive
  // live frames interpolate against each other (a live feed has no "next
  // keyframe", so prev === target would otherwise freeze the avatar).
  const liveSeqRef = useRef(-1);
  const livePrevPoseRef = useRef<RetargetedPose | null>(null);
  const liveWasActiveRef = useRef(false);
  // Mirror of the `liveActive` prop. The animation loop and the auto-advance
  // interval are both installed once, so they would otherwise capture a stale
  // `false` forever and the live toggle would appear to do nothing.
  const liveActiveRef = useRef(liveActive);
  liveActiveRef.current = liveActive;

  const [playbackSpeed, setPlaybackSpeed] = useState(externalSpeed);
  const [isPlaying, setIsPlaying] = useState(externalPlaying);
  const [frameIndex, setFrameIndex] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [loadingError, setLoadingError] = useState<string | null>(null);

  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    if (!modelReady) return;
    if (cameraRef.current) cameraRef.current.position.set(0, 1.5, 1.8);
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 1.5, 0);
      controlsRef.current.update();
    }
  }, [modelReady]);

  const playbackSpeedRef = useRef(externalSpeed);
  const isPlayingRef = useRef(externalPlaying);
  playbackSpeedRef.current = playbackSpeed;
  isPlayingRef.current = isPlaying;

  const effectiveFrameIndex = externalFrameIndex !== undefined ? externalFrameIndex : frameIndex;

  useEffect(() => {
    onModelReady?.(modelReady);
  }, [modelReady, onModelReady]);

  const loadXbot = useCallback((scene: THREE.Scene) => {
    return new Promise<{ model: THREE.Group; bones: Record<string, THREE.Bone> }>((resolve, reject) => {
      const loader = new GLTFLoader();
      loader.load(
        "/models/Xbot.glb",
        (gltf) => {
          try {
            const model = gltf.scene;
            model.position.y = 0;
            const bones: Record<string, THREE.Bone> = {};
            const unmapped: string[] = [];
            const allBoneNames: string[] = [];
            const boneCountByType: Record<string, number> = { bone: 0, group: 0, other: 0 };
            model.traverse((child) => {
              if (child.name) {
                allBoneNames.push(child.name);
                if ((child as THREE.Bone).isBone) {
                  boneCountByType.bone++;
                } else if (child instanceof THREE.Group) {
                  boneCountByType.group++;
                } else {
                  boneCountByType.other++;
                }
              }
              if ((child as THREE.Bone).isBone && child.name) {
                let simpleName = child.name.replace(/mixamorig:?/, "");
                simpleName = simpleName.replace(/\.\d+$/, "");
                const mapped = BONE_MAP[simpleName];
                if (mapped) {
                  bones[mapped] = child as THREE.Bone;
                } else {
                  unmapped.push(child.name);
                }
              } else if (child.type === "Bone" && child.name) {
                let simpleName = child.name.replace(/mixamorig:?/, "");
                simpleName = simpleName.replace(/\.\d+$/, "");
                const mapped = BONE_MAP[simpleName];
                if (mapped) {
                  bones[mapped] = child as THREE.Bone;
                } else {
                  unmapped.push(child.name);
                }
              }
            });

            console.log("[avatar] bone types:", boneCountByType, "total named:", allBoneNames.length);

            const boneToAvatarKey: Record<string, string> = {};
            for (const [avKey, bone] of Object.entries(bones)) {
              boneToAvatarKey[bone.name] = avKey;
            }

            if (Object.keys(bones).length === 0) {
              console.error(
                "[avatar] NO BONES MAPPED! BONE_MAP keys:",
                Object.keys(BONE_MAP).join(", "),
                "GLTF bones found:",
                allBoneNames.slice(0, 20).join(", "),
                boneCountByType,
              );
            } else {
              console.log(
                "[avatar] mapped " + Object.keys(bones).length + " bones:",
                Object.keys(bones).join(", "),
              );
            }
            if (unmapped.length > 0) {
              console.log("[avatar] unmapped bones:", unmapped.join(", "));
            }
            allBoneNamesRef.current = allBoneNames;

            model.traverse((child) => {
              const mesh = child as THREE.Mesh;
              mesh.castShadow = true;
              mesh.receiveShadow = true;
            });

            (window as any).__avatarDebug = {
              get bones() { return Object.keys(bones); },
              get boneToAvatarKey() { return boneToAvatarKey; },
              get allBoneNames() { return allBoneNamesRef.current; },
              get boneCountByType() { return boneCountByType; },
              get isAnimating() { return isAnimatingRef.current; },
              get isPlaying() { return isPlayingRef.current; },
              get targetFrame() { return targetFrameRef2.current; },
              get frameIndex() { return effectiveFrameIndex; },
              get totalFrames() { return totalFrames; },
              get hasAvatar() { return !!avatarRef.current; },
              get currentFramesCount() { return currentFramesRef.current.length; },
              get bonesMapped() { return Object.keys(bones).length; },
              get bonesBound() {
                const frames = currentFramesRef.current;
                if (frames.length === 0) return 0;
                const cur = frames[0];
                let count = 0;
                for (const key of AVATAR_KEYS) {
                  if (bones[key] && cur[key]) count++;
                }
                return count;
              },
              get missingBones() {
                const result: string[] = [];
                for (const key of AVATAR_KEYS) {
                  if (!bones[key]) result.push(key);
                }
                return result;
              },
              get currentRotations() {
                const av = avatarRef.current;
                if (!av) return null;
                const result: Record<string, [number, number, number] | null> = {};
                for (const key of AVATAR_KEYS) {
                  const bone = av.bones[key];
                  if (bone) {
                    result[key] = [bone.rotation.x, bone.rotation.y, bone.rotation.z];
                  }
                }
                return result;
              },
            };

              // Capture bind pose quaternions for quaternion bones
              const capturedBindPose: Record<string, THREE.Quaternion> = {};
              for (const key of QUAT_BONE_KEYS) {
                const bone = bones[key];
                if (bone) {
                  capturedBindPose[key] = bone.quaternion.clone();
                }
              }
              bindPoseQuats.current = capturedBindPose;

              // Capture parent world quaternions at bind pose for arm bone retargeting
              model.updateMatrixWorld();
              const parentBoneKeys = [
                "torso", "neck",
                "leftShoulder", "rightShoulder",
                "leftUpperArm", "rightUpperArm",
                "leftForearm", "rightForearm",
              ];
              const capturedParentWorld: Record<string, THREE.Quaternion> = {};
              for (const key of parentBoneKeys) {
                const bone = bones[key];
                if (bone) {
                  const worldQuat = new THREE.Quaternion();
                  bone.getWorldQuaternion(worldQuat);
                  capturedParentWorld[key] = worldQuat;
                }
              }
              parentWorldQuatsRef.current = capturedParentWorld;

              scene.add(model);

              // Resolve skeleton using boneResolver for the new retargeting pipeline
              try {
                resolvedSkeletonRef.current = resolveSkeleton(model);
                console.log("[avatar] skeleton resolved:", resolvedSkeletonRef.current.boneOrder);
              } catch (e) {
                console.warn("[avatar] skeleton resolution failed:", e);
              }

              resolve({ model, bones });
          } catch (err) {
            reject(err);
          }
        },
        undefined,
        reject,
      );
     });
  }, []);

  const toParentWorldQuats = useCallback((): ParentWorldQuats => {
    const result: ParentWorldQuats = {};
    for (const [key, q] of Object.entries(parentWorldQuatsRef.current)) {
      result[key] = [q.x, q.y, q.z, q.w];
    }
    return result;
  }, []);

  const [signSequenceState, setSignSequenceState] = useState(signSequence);
  useEffect(() => {
    setSignSequenceState(signSequence);
  }, [signSequence]);

  const loadSignData = useCallback(async (label: string, parentWorldQuats?: ParentWorldQuats) => {
    if (!hasLabelData(label)) {
      console.log(`[avatar] loadSignData: ${label} NOT available`);
      return;
    }

     console.log(`[avatar] loadSignData: ${label} OK`);

      // Try new CanonicalPose pipeline first (SMPL-X adapter)
      const skeleton = resolvedSkeletonRef.current;
      let usedCanonical = false;
      usedCanonicalRef.current = false;

     if (skeleton) {
      // PHASE 14: prefer the RAW MediaPipe recordings in /avatar-data/.
      //
      // The SMPL-X derivatives in /avatar-data-smplx/ are still corrupt: the
      // exporter that produced them omitted the elbow joint, so every arm was
      // stored ~4x too long and the retargeter folded them into the torso. The
      // Python exporter is fixed, but the 402 files cannot be regenerated in
      // this environment, so we play the original recordings instead. They feed
      // the SAME mediapipeToCanonicalPose() the live webcam uses -- the path
      // covered by all 37 verify:retarget tests and the bit-for-bit golden
      // snapshot -- so this is a data-source swap, not a change to any maths.
      //
      // SMPL-X stays as a fallback, so a clip with no raw MediaPipe file does
      // not become "unavailable" purely because of this swap.
      const canonicalDataset =
        (await loadLabelDataMediaPipe(label)) ?? (await loadLabelDataCanonical(label));
       if (canonicalDataset && canonicalDataset.frames.length > 0) {
         const step = Math.max(1, Math.floor(canonicalDataset.frames.length / 16));
         const sampled: (CanonicalPose | null)[] = [];
         for (let i = 0; i < canonicalDataset.frames.length; i += step) {
           sampled.push(canonicalDataset.frames[i]);
         }
         if (sampled.length > 0 && sampled[sampled.length - 1] !== canonicalDataset.frames[canonicalDataset.frames.length - 1]) {
           sampled.push(canonicalDataset.frames[canonicalDataset.frames.length - 1]);
         }
         if (sampled.length < 2) {
           sampled.push(canonicalDataset.frames[0]);
         }

         // Retarget each canonical pose to avatar bone rotations
         const retargeted: RetargetedPose[] = [];
         for (const pose of sampled) {
           if (pose) {
             const r = retarget(skeleton, pose);
             retargeted.push(r);
           } else {
             retargeted.push(retarget(skeleton, {
               body: {
                 hipsPos: [0, 0, 0],
                 shoulderHeight: 0,
                 spineDir: [0, 1, 0],
                 shoulderAxis: [0, 0, 1],
                 leftShoulder: [0, 0, 0], leftElbow: [0, 0, 0], leftWrist: [0, 0, 0],
                 rightShoulder: [0, 0, 0], rightElbow: [0, 0, 0], rightWrist: [0, 0, 0],
                 neckPos: [0, 0, 0], headPos: [0, 0, 0], headDir: [0, 1, 0],
                 leftHip: [0, 0, 0], leftKnee: [0, 0, 0], leftAnkle: [0, 0, 0],
                 rightHip: [0, 0, 0], rightKnee: [0, 0, 0], rightAnkle: [0, 0, 0],
               } as any,
               leftHand: { wristPos: [0, 0, 0], palmNormal: [0, 0, 1], palmDir: [0, 0, 1], fingers: {} } as any,
               rightHand: { wristPos: [0, 0, 0], palmNormal: [0, 0, 1], palmDir: [0, 0, 1], fingers: {} } as any,
               timestamp: 0,
             } as CanonicalPose));
           }
         }

          currentRetargetedRef.current = retargeted;
          currentFramesRef.current = [];
          prevFrameRef.current = null;
          targetFrameRef2.current = null;
          prevRetargetedRef.current = retargeted[0];
          targetRetargetedRef.current = retargeted[Math.min(1, retargeted.length - 1)];
          setFrameIndex(0);
          setTotalFrames(retargeted.length);
          transitionProgressRef.current = 0;
          transitionStartTimeRef.current = timeRef.current;
          isAnimatingRef.current = true;
          usedCanonical = true;
          usedCanonicalRef.current = true;

          // Apply initial pose immediately
          const av = avatarRef.current;
          if (skeleton && retargeted[0] && av) {
            applyBoneRotations(retargeted[0].boneRotations, skeleton);
            if (av.model) {
              if (APPLY_ROOT_TRANSLATION) {
                // PHASE 7: new sequence -> the first frame becomes the baseline,
                // so the root starts at the origin and only MOVES relative to it.
                rootSolverRef.current.reset();
                rootSolverRef.current.solve(retargeted[0].hipsWorldPos, tmpRootOffset.current);
                av.model.position.copy(tmpRootOffset.current);
              } else {
                av.model.position.set(0, 0, 0);
              }
            }
          }

          console.log(`[avatar] loadSignData: canonical path loaded ${retargeted.length} retargeted frames`);
        }
        onFrameChange?.(0, currentRetargetedRef.current.length);
        onSignChange?.(label);
      }

     // Fallback: traditional quaternion-based path
     if (!usedCanonical) {
     const dataset = await loadLabelData(label, parentWorldQuats);
     if (!dataset || dataset.frames.length === 0) {
      console.warn(`[avatar] loadSignData: ${label} has no frames`);
      return;
    }

    const frames = dataset.frames;
    const step = Math.max(1, Math.floor(frames.length / 16));
    const sampled: AvatarFrame[] = [];
    for (let i = 0; i < frames.length; i += step) {
      sampled.push(frames[i]);
    }
    if (sampled.length > 0 && sampled[sampled.length - 1] !== frames[frames.length - 1]) {
      sampled.push(frames[frames.length - 1]);
    }
    if (sampled.length < 2) {
      sampled.push(frames[0]);
    }

    currentFramesRef.current = sampled;
    setFrameIndex(0);
    setTotalFrames(sampled.length);
    prevFrameRef.current = sampled[0];
    targetFrameRef2.current = sampled[Math.min(1, sampled.length - 1)];
    transitionProgressRef.current = 0;
    transitionStartTimeRef.current = timeRef.current;
    isAnimatingRef.current = true;

    const av = avatarRef.current;
    if (av) {
      const mappedBones = Object.keys(av.bones);
      let nonZero = 0;
      const zeroBones: string[] = [];
      for (const key of AVATAR_KEYS) {
        if (av.bones[key] && sampled[0][key]) {
          const v = sampled[0][key];
          if (v[0] !== 0 || v[1] !== 0 || v[2] !== 0) nonZero++;
          else zeroBones.push(key);
        } else {
          zeroBones.push(key + "(no bone/frame)");
        }
      }
      console.log(`[avatar] loadSignData: ${mappedBones.length} bones mapped, ${nonZero} non-zero in frame0, zero: ${zeroBones.join(", ") || "none"}`);
    }

    onFrameChange?.(0, sampled.length);
    onSignChange?.(label);
      } // end if (!usedCanonical)
  }, [onFrameChange, onSignChange]);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050810);
    scene.fog = new THREE.FogExp2(0x050810, 0.06);

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 1.5, 1.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.5;
    controls.maxDistance = 4;
    controls.maxPolarAngle = Math.PI * 0.7;
    controls.minPolarAngle = Math.PI * 0.15;
    controls.target.set(0, 1.5, 0);
    controlsRef.current = controls;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0.5;
    controls.enablePan = false;
    controls.enableRotate = false;
    controls.enableZoom = false;
    controls.update();

    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTexture;

    scene.add(new THREE.AmbientLight(0x404060, 1.0));

    const keyLight = new THREE.DirectionalLight(0xffeedd, 1.5);
    keyLight.position.set(3, 4, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 15;
    keyLight.shadow.camera.left = -2;
    keyLight.shadow.camera.right = 2;
    keyLight.shadow.camera.top = 2;
    keyLight.shadow.camera.bottom = -2;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.radius = 4;
    scene.add(keyLight);

    const spotLight = new THREE.SpotLight(0x00ddff, 0.4, 12, Math.PI / 6, 0.5, 1.5);
    spotLight.position.set(-2, 4, 2);
    spotLight.target.position.set(0, -0.5, 0);
    scene.add(spotLight);
    scene.add(spotLight.target);

    const spotLight2 = new THREE.SpotLight(0xdd44ff, 0.3, 12, Math.PI / 7, 0.5, 1.5);
    spotLight2.position.set(2.5, 3, -1);
    spotLight2.target.position.set(0, -0.3, 0);
    scene.add(spotLight2);
    scene.add(spotLight2.target);

    const fillLight = new THREE.PointLight(0xffffff, 0.3, 10);
    fillLight.position.set(2.5, 2, 1.5);
    scene.add(fillLight);

    const rimLight = new THREE.PointLight(0xffeedd, 0.4, 10);
    rimLight.position.set(-2.5, -0.5, -2);
    scene.add(rimLight);

    const backLight = new THREE.PointLight(0x88ccff, 0.3, 10);
    backLight.position.set(0, 1.5, -3);
    scene.add(backLight);

    // PHASE 25 — hand/shoulder accent lights dimmed and de-saturated.
    //
    // These four coloured point lights sat at fixed world positions near the
    // hands and shoulders with intensity 0.3 / 0.2. They are the dominant
    // coloured light on the hands: a saturated green (0x00ffcc) pool directly
    // on the signing hands, which both raised the hands above the bloom
    // threshold and destroyed the skin-tone contrast that makes a handshape
    // readable. Halved to 0.15 / 0.1 and pulled back toward neutral so the
    // hands read as skin, not as emissive props.
    const glowHandL = new THREE.PointLight(0x00ffcc, 0.15, 2);
    glowHandL.position.set(0.6, -1.2, 0.3);
    scene.add(glowHandL);
    const glowHandR = new THREE.PointLight(0x00ffcc, 0.15, 2);
    glowHandR.position.set(-0.6, -1.2, 0.3);
    scene.add(glowHandR);
    const glowShoulderL = new THREE.PointLight(0xff44aa, 0.1, 2);
    glowShoulderL.position.set(0.3, 0.3, 0.2);
    scene.add(glowShoulderL);
    const glowShoulderR = new THREE.PointLight(0xff44aa, 0.1, 2);
    glowShoulderR.position.set(-0.3, 0.3, 0.2);
    scene.add(glowShoulderR);

    const particleCount = 200;
    const particleGeo = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      particlePositions[i * 3] = (Math.random() - 0.5) * 6;
      particlePositions[i * 3 + 1] = Math.random() * 3 - 1;
      particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 6;
    }
    particleGeo.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: 0x00ddff, size: 0.01, transparent: true, opacity: 0.2,
      sizeAttenuation: true, depthWrite: false,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    const loadAndAnimate = async () => {
      try {
        resetAvatarData();
        const labels = await loadAvatarData();
        allAvailableRef.current = labels;

        const modelLoaded = await loadXbot(scene).catch((err) => {
          console.error("[avatar] Xbot load failed:", err);
          throw err;
        });
        avatarRef.current = modelLoaded;
        modelReadyRef.current = true;
        setModelReady(true);
        setLoadingError(null);

        const parentWorldQuatsForRetarget = toParentWorldQuats();

        const checkInterval = setInterval(async () => {
          // PHASE 10: never swap in a new recorded clip while the live webcam
          // is driving the avatar, or the two sources would alternate and the
          // pose would visibly stutter between them.
          //
          // Read through a ref, NOT the `liveActive` prop: adding it to this
          // effect's dependency array would tear down and rebuild the entire
          // WebGLRenderer + GLTF scene on every live toggle, which is precisely
          // the leak this phase is meant to close. A ref keeps the toggle free.
          if (liveActiveRef.current) return;
          if (signSequenceState && signSequenceState.length > 0) {
            const now = Date.now();
            if (now - lastSignTimeRef.current > 12000) {
              lastSignTimeRef.current = now;
              signIndexRef.current = (signIndexRef.current + 1) % signSequenceState.length;
              await loadSignData(signSequenceState[signIndexRef.current], parentWorldQuatsForRetarget);
            }
          }
        }, 500);

        if (signSequenceState && signSequenceState.length > 0) {
          lastSignTimeRef.current = Date.now();
          await loadSignData(signSequenceState[0], parentWorldQuatsForRetarget);
        }

        return () => clearInterval(checkInterval);
      } catch (err) {
        console.error("[avatar] loadAndAnimate failed:", err);
        setLoadingError(err instanceof Error ? err.message : "Failed to load 3D avatar");
        setModelReady(false);
      }
    };

    let isRunning = true;

    const animate = () => {
      if (!isRunning) return;
      animationIdRef.current = requestAnimationFrame(animate);
      timeRef.current += 0.016;

    // PHASE 10: per-frame console logging removed. `console.log` on the render
    // path costs real time at 60fps (each call formats and pushes to the devtools
    // buffer) and was the single largest source of jank in the live pipeline.
    // These blocks were unconditional for the first 500 frames.

      const av = avatarRef.current;
      if (!av) return;

      const t = timeRef.current;
      breathPhaseRef.current = t * 1.2;
      idleWeightRef.current = Math.sin(t * 0.7) * 0.5 + 0.5;

      // PHASE 10 — LIVE SOURCE. Consume a new webcam frame, if one is pending.
      //
      // Placement is deliberate: this runs BEFORE the recorded-playback gate and
      // takes priority over it, so the two sources can never fight over the same
      // bones in one frame. The live pose is pushed through the SAME
      // prev/target retargeted refs the recorded path uses, so everything
      // downstream (interpolation, root solver, footing, rendering) is shared
      // and unchanged. That is what makes the toggle seamless: switching source
      // changes only which pose lands in the refs, never the code that consumes
      // it, and the existing SMOOTHING slerp below interpolates across the
      // switch instead of snapping.
      const liveFrame = liveFrameRef?.current ?? null;
      if (liveActiveRef.current && liveFrame && liveFrame.seq !== liveSeqRef.current) {
        const skel = resolvedSkeletonRef.current;
        if (skel) {
          liveSeqRef.current = liveFrame.seq;
          // Same adapter the recorded SMPL-X path and verify:browser use.
          const livePose = mediapipeToCanonicalPose(liveFrame.vector, 0);
          if (livePose) {
            const liveRet = retarget(skel, livePose);
            // Reset the root baseline on the first live frame, otherwise the
            // delta solver would jump the model by the difference between a
            // recorded clip's hips and the live hips.
            if (!liveWasActiveRef.current) {
              liveWasActiveRef.current = true;
              livePrevPoseRef.current = null;
              if (APPLY_ROOT_TRANSLATION) rootSolverRef.current.reset();
            }
            // Interpolate from the previous live frame; on the very first frame
            // there is no previous, so use the pose itself (a no-op slerp) and
            // let SMOOTHING ease the bones in from wherever they currently are.
            prevRetargetedRef.current = livePrevPoseRef.current ?? liveRet;
            livePrevPoseRef.current = liveRet;
            targetRetargetedRef.current = liveRet;
            usedCanonicalRef.current = true;
            isAnimatingRef.current = true;
            transitionProgressRef.current = 1;
          }
        }
      }
      if (!liveActiveRef.current && liveWasActiveRef.current) {
        // Camera disabled: hand control back to recorded playback and drop the
        // live baseline so re-enabling starts a fresh root baseline.
        liveWasActiveRef.current = false;
        livePrevPoseRef.current = null;
        liveSeqRef.current = -1;
      }

      // PHASE 10 BUGFIX: this gate used to require `targetFrameRef2.current`,
      // but the CANONICAL loader (loadSignData) sets `targetFrameRef2.current =
      // null` and drives playback through `targetRetargetedRef` instead. So for
      // every canonical clip - i.e. all of them, since SMPL-X data is the only
      // source - the condition was false and the whole interpolation block was
      // skipped. The avatar sat at the frame-0 pose applied by
      // applyBoneRotations() and never animated. The test suite missed it
      // because verify:browser drives a standalone harness that re-implements
      // this maths rather than mounting the component.
      // Fix: gate on whichever target ref the ACTIVE source uses.
      const gateTarget = usedCanonicalRef.current
        ? targetRetargetedRef.current
        : targetFrameRef2.current;
      if (isAnimatingRef.current && gateTarget && isPlayingRef.current) {
        const elapsed = t - transitionStartTimeRef.current;
        const duration = transitionDuration / playbackSpeedRef.current;
        const rawT = clamp(elapsed / duration, 0, 1);
        transitionProgressRef.current = easeInOutCubic(rawT);

        if (usedCanonicalRef.current) {
          // Canonical path: interpolate RetargetedPose using applyBoneRotations + slerp on quaternions
          const curRet = prevRetargetedRef.current;
          const tgtRet = targetRetargetedRef.current;
          const progress = transitionProgressRef.current;

          if (curRet && tgtRet) {
            const av = avatarRef.current;
            const skeleton = resolvedSkeletonRef.current;
             if (av && skeleton) {
              // Interpolate root position
              tmpVec1.current.copy(curRet.hipsWorldPos);
              tmpVec2.current.copy(tgtRet.hipsWorldPos);
              tmpVec1.current.lerp(tmpVec2.current, progress);
              if (av.model) {
                if (APPLY_ROOT_TRANSLATION) {
                  // PHASE 7: feed the INTERPOLATED hips through the delta
                  // solver, so the envelope/slew/floor clamps apply every frame
                  // instead of once per keyframe.
                  rootSolverRef.current.solve(tmpVec1.current, tmpRootOffset.current);
                  av.model.position.copy(tmpRootOffset.current);
                } else {
                  av.model.position.set(0, 0, 0);
                }
              }

              // Interpolate all bone rotations by slerping quaternions.
              // PHASE 10: this used to build `new Set<string>()` and two
              // closures every frame just to union the two bone-name maps.
              // Both maps are keyed by the SAME bone set (retarget() emits an
              // identical key set for every pose), so iterating `curRet` alone
              // is equivalent -- and the reusable module-level Set below keeps
              // it correct even if a key were ever missing from one side.
              const allBoneNames = boneNameScratch;
              allBoneNames.clear();
              curRet.boneRotations.forEach((_, name) => allBoneNames.add(name));
              tgtRet.boneRotations.forEach((_, name) => allBoneNames.add(name));

              for (const boneName of allBoneNames) {
                const curRot = curRet.boneRotations.get(boneName);
                const tgtRot = tgtRet.boneRotations.get(boneName);
                const bone = skeleton.bones.get(boneName)?.bone;
                if (curRot && tgtRot && bone) {
                  resultQuat.current.slerpQuaternions(curRot, tgtRot, progress);
                  bone.quaternion.slerp(resultQuat.current, SMOOTHING);
                  bone.quaternion.normalize();
                }
              }

              // PHASE 9: additive ground contact, applied AFTER the interpolated
              // retargeted pose (not after applyBoneRotations — this path slerps
              // between keyframes rather than re-applying the bone map). The
              // solver reads the current world state and rewrites ONLY the four
              // leg bones' local quaternions; the ankle keeps its bind rotation
              // and the root offset above is untouched, so this cannot cheat by
              // lifting the model. Bounded to a corrective pass and clamped so a
              // fully extended leg can never produce NaN.
              if (av.model) {
                av.model.updateMatrixWorld(true);
                solveFooting(skeleton, { floorY: FOOTING_FLOOR_Y });
                av.model.updateMatrixWorld(true);
              }
            }
          }
        } else {
          // Traditional path: interpolate AvatarFrame arrays
          const cur = prevFrameRef.current;
          const tgt = targetFrameRef2.current;

        if (cur && tgt) {
          let bonesUpdated = 0;
          let bonesSkipped = 0;
            for (const key of AVATAR_KEYS) {
              const bone = av.bones[key];
              const p1 = cur[key] as number[];
              const p2 = tgt[key] as number[];
              if (bone && p1 && p2) {
                const isQuatBone = QUAT_BONE_KEYS.has(key);
                   if (isQuatBone && p1.length >= 4 && p2.length >= 4) {
                     const isZero =
                       Math.abs(p1[0]) < 1e-8 && Math.abs(p1[1]) < 1e-8 &&
                       Math.abs(p1[2]) < 1e-8 && Math.abs(p1[3]) < 1e-8;
                     if (isZero) { bonesSkipped++; continue; }

                     tmpQuat1.current.set(p1[0], p1[1], p1[2], p1[3]);
                     tmpQuat2.current.set(p2[0], p2[1], p2[2], p2[3]);
                     resultQuat.current.slerpQuaternions(
                       tmpQuat1.current, tmpQuat2.current,
                       transitionProgressRef.current,
                     );
                     resultQuat.current.normalize();

                     const targetQuat = resultQuat.current;
                     const dot = bone.quaternion.dot(targetQuat);
                     const angle = Math.acos(Math.min(1, Math.max(-1, Math.abs(dot)))) * 2;

                     if (angle > MAX_BONE_ANGLE) {
                       bone.quaternion.slerp(targetQuat, SMOOTHING * 0.5);
                     } else {
                       bone.quaternion.slerp(targetQuat, SMOOTHING);
                     }
                     bone.quaternion.normalize();
                     bonesUpdated++;
                } else {
                  const isZero =
                    Math.abs(p1[0]) < 1e-6 && Math.abs(p1[1]) < 1e-6 && Math.abs(p1[2]) < 1e-6;
                  if (isZero) { bonesSkipped++; continue; }
                  // Position bones (root, hips) - interpolate position, not rotation
                  const t = transitionProgressRef.current;
                  bone.position.set(
                    p1[0] + (p2[0] - p1[0]) * t,
                    p1[1] + (p2[1] - p1[1]) * t,
                    p1[2] + (p2[2] - p1[2]) * t,
                  );
                  bonesUpdated++;
                }
              } else {
                bonesSkipped++;
              }
            }
        }
        } // end else (traditional path)

         if (rawT >= 1) {
          isAnimatingRef.current = false;
          if (usedCanonicalRef.current) {
            prevRetargetedRef.current = targetRetargetedRef.current;
            const retargeted = currentRetargetedRef.current;
            const curIdx = retargeted.findIndex((f) => f === prevRetargetedRef.current);
            const nextIdx = curIdx >= 0 ? curIdx + 1 : 1;

            if (nextIdx < retargeted.length) {
              targetRetargetedRef.current = retargeted[nextIdx];
              transitionProgressRef.current = 0;
              transitionStartTimeRef.current = t;
              isAnimatingRef.current = true;
              setFrameIndex(curIdx);
              onFrameChange?.(curIdx, retargeted.length);
            } else {
              prevRetargetedRef.current = retargeted[0];
              targetRetargetedRef.current = retargeted[Math.min(1, retargeted.length - 1)];
              transitionProgressRef.current = 0;
              transitionStartTimeRef.current = t;
              isAnimatingRef.current = true;
              setFrameIndex(0);
              onFrameChange?.(0, retargeted.length);
            }
          } else {
            prevFrameRef.current = targetFrameRef2.current;
            const frames = currentFramesRef.current;
            const curIdx = frames.findIndex((f) => f === prevFrameRef.current);
            const nextIdx = curIdx >= 0 ? curIdx + 1 : 1;

            if (nextIdx < frames.length) {
              targetFrameRef2.current = frames[nextIdx];
              transitionProgressRef.current = 0;
              transitionStartTimeRef.current = t;
              isAnimatingRef.current = true;
              setFrameIndex(curIdx);
              onFrameChange?.(curIdx, frames.length);
            } else {
              prevFrameRef.current = frames[0];
              targetFrameRef2.current = frames[Math.min(1, frames.length - 1)];
              transitionProgressRef.current = 0;
              transitionStartTimeRef.current = t;
              isAnimatingRef.current = true;
              setFrameIndex(0);
              onFrameChange?.(0, frames.length);
            }
          }
        }
      } else if (isPlayingRef.current && !isAnimatingRef.current) {
        // PHASE 8: `!isAnimatingRef.current` is REQUIRED here, not cosmetic.
        // The retarget branch above only runs when isAnimating AND
        // targetFrameRef2 are both set, so a frame where sign data is loaded
        // (isAnimating true) but targetFrameRef2 is momentarily null -- e.g.
        // right after a seek or on the first frame of a load -- previously fell
        // through to this branch and OVERWROTE head, neck, torso and BOTH arms
        // and BOTH hands with idle rotations. That is exactly the "destructive
        // override" the idle layer must never do: with the guard, such a frame
        // simply holds the last retargeted pose, which is safe and correct.
        const sway = idleWeightRef.current;
        if (av.bones.head) {
          const bp = bindPoseQuats.current["head"];
          if (bp) {
            tmpEuler1.current.set(Math.sin(t * 0.5) * 0.03, Math.sin(t * 0.35) * 0.06 + sway * 0.03, 0);
            tmpQuat1.current.setFromEuler(tmpEuler1.current);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.head.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.head.quaternion.normalize();
          }
        }
        if (av.bones.neck) {
          const bp = bindPoseQuats.current["neck"];
          if (bp) {
            tmpEuler1.current.set(0, 0, Math.sin(t * 0.4) * 0.015);
            tmpQuat1.current.setFromEuler(tmpEuler1.current);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.neck.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.neck.quaternion.normalize();
          }
        }

        av.model.position.y = Math.sin(t * 0.6) * 0.008;
        av.model.rotation.z = Math.sin(t * 0.3) * 0.015 * sway;

        if (av.bones.leftShoulder) {
          const bp = bindPoseQuats.current["leftShoulder"];
          if (bp) {
            tmpEuler1.current.set(0, 0, -0.03 + Math.sin(t * 0.5) * 0.02);
            tmpQuat1.current.setFromEuler(tmpEuler1.current);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.leftShoulder.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.leftShoulder.quaternion.normalize();
          }
        }
        if (av.bones.rightShoulder) {
          const bp = bindPoseQuats.current["rightShoulder"];
          if (bp) {
            tmpEuler1.current.set(0, 0, 0.03 - Math.sin(t * 0.5) * 0.02);
            tmpQuat1.current.setFromEuler(tmpEuler1.current);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.rightShoulder.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.rightShoulder.quaternion.normalize();
          }
        }
        if (av.bones.leftUpperArm) {
          const bp = bindPoseQuats.current["leftUpperArm"];
          if (bp) {
            idleAxis.current.set(1, 0, 0);
            tmpQuat1.current.setFromAxisAngle(idleAxis.current, Math.sin(t * 0.4) * 0.01);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.leftUpperArm.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.leftUpperArm.quaternion.normalize();
          }
        }
        if (av.bones.rightUpperArm) {
          const bp = bindPoseQuats.current["rightUpperArm"];
          if (bp) {
            idleAxis.current.set(1, 0, 0);
            tmpQuat1.current.setFromAxisAngle(idleAxis.current, Math.sin(t * 0.4 + 0.5) * 0.01);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.rightUpperArm.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.rightUpperArm.quaternion.normalize();
          }
        }
        if (av.bones.leftForearm) {
          const bp = bindPoseQuats.current["leftForearm"];
          if (bp) {
            idleAxis.current.set(1, 0, 0);
            tmpQuat1.current.setFromAxisAngle(idleAxis.current, Math.sin(t * 0.6 + 1.0) * 0.015);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.leftForearm.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.leftForearm.quaternion.normalize();
          }
        }
        if (av.bones.rightForearm) {
          const bp = bindPoseQuats.current["rightForearm"];
          if (bp) {
            idleAxis.current.set(1, 0, 0);
            tmpQuat1.current.setFromAxisAngle(idleAxis.current, Math.sin(t * 0.6 + 1.5) * 0.015);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.rightForearm.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.rightForearm.quaternion.normalize();
          }
        }
        if (av.bones.leftHand) {
          const bp = bindPoseQuats.current["leftHand"];
          if (bp) {
            idleAxis.current.set(1, 0, 0);
            tmpQuat1.current.setFromAxisAngle(idleAxis.current, Math.sin(t * 0.5 + 1.0) * 0.01);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.leftHand.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.leftHand.quaternion.normalize();
          }
        }
        if (av.bones.rightHand) {
          const bp = bindPoseQuats.current["rightHand"];
          if (bp) {
            idleAxis.current.set(1, 0, 0);
            tmpQuat1.current.setFromAxisAngle(idleAxis.current, Math.sin(t * 0.5 + 1.5) * 0.01);
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.rightHand.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.rightHand.quaternion.normalize();
          }
        }
        if (av.bones.torso) {
          const bp = bindPoseQuats.current["torso"];
          if (bp) {
            tmpQuat1.current.setFromEuler(tmpEuler1.current.set(Math.sin(t * 0.3 + 2.0) * 0.005, 0, Math.sin(t * 0.25) * 0.008));
            resultQuat.current.multiplyQuaternions(tmpQuat1.current, bp);
            av.bones.torso.quaternion.slerp(resultQuat.current, 0.2);
            av.bones.torso.quaternion.normalize();
          }
        }
      }

      // Normalize all bone quaternions to prevent drift
      if (avatarRef.current?.bones) {
        for (const bone of Object.values(avatarRef.current.bones)) {
            bone.quaternion.normalize();
          }
      }

      try {
        controls.update();
        const pp = particles.geometry.attributes.position.array as Float32Array;
        for (let i = 0; i < particleCount; i++) {
          pp[i * 3 + 1] += 0.003;
          if (pp[i * 3 + 1] > 2) pp[i * 3 + 1] = -1;
        }
        particles.geometry.attributes.position.needsUpdate = true;
        composer.render();
      } catch (e) {
        console.error('[avatar] render error:', e);
      }
    };

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    // PHASE 25 — bloom reduced for sign-language readability.
    //
    // Was (strength, radius, threshold) = (0.25, 0.3, 0.85). The threshold was
    // low enough that the lit skin of the hands and forearms crossed it, so the
    // bloom pass bloomed the SUBJECT rather than the background rim, wrapping
    // the hands in a halo exactly when they cross the torso and readability
    // matters most.
    //
    // New values are chosen so bloom only touches genuine highlights (the rim
    // and the accent lights) and leaves lit skin below threshold:
    //   strength   0.25 -> 0.10  (halo was flooding the silhouette edges)
    //   radius     0.30 -> 0.20  (tighter, so glow stays local to highlights)
    //   threshold  0.85 -> 0.92  (only near-white pixels bloom now)
    // ACESFilmic tone mapping is kept: it compresses highlights gracefully and
    // is what lets the key light read without clipping.
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height),
      0.10, 0.2, 0.92
    );
    composer.addPass(bloomPass);

    composer.addPass(new OutputPass());

    const resizeObserver = new ResizeObserver(() => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 400;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      composer.setSize(w, h);
      bloomPass.setSize(w, h);
    });
    resizeObserver.observe(container);

    const onWindowResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 400;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      composer.setSize(w, h);
      bloomPass.setSize(w, h);
    };
    window.addEventListener("resize", onWindowResize);

    controls.addEventListener("start", () => { controls.autoRotate = false; });
    animate();

    const cleanupPromise = loadAndAnimate();

    return () => {
      isRunning = false;
      cancelAnimationFrame(animationIdRef.current);
      resizeObserver.disconnect();
      window.removeEventListener("resize", onWindowResize);
      cleanupPromise.then(fn => fn?.());
      if (avatarRef.current) {
        avatarRef.current.model.traverse((child) => {
          const m = child as THREE.Mesh;
          if (m.isMesh) {
            m.geometry?.dispose();
            const mat = m.material;
            const mats = Array.isArray(mat) ? mat : [mat];
            for (let i = 0; i < mats.length; i++) {
              const mm = mats[i];
              if (!mm) continue;
              // PHASE 10: material.dispose() does NOT release the textures the
              // material references -- the GL texture stays resident on the GPU
              // until it is disposed explicitly. Xbot's skin/baseColor/normal
              // maps were leaking for the lifetime of the context.
              for (const key in mm) {
                const value = (mm as unknown as Record<string, unknown>)[key];
                if (value && (value as THREE.Texture).isTexture) {
                  (value as THREE.Texture).dispose();
                }
              }
              mm.dispose();
            }
          }
        });
        scene.remove(avatarRef.current.model);
        avatarRef.current = null;
      }
      // PHASE 10: the particle system is scene-owned, not avatar-owned, so the
      // traverse above never reached it. It was leaking a geometry + material
      // (and their GPU buffers) on every unmount.
      particleGeo.dispose();
      particleMat.dispose();
      scene.remove(particles);
      // Post-processing passes own render targets; dispose them with the
      // composer or their framebuffers leak.
      bloomPass.dispose();
      composer.dispose();
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [signSequenceState, loadSignData]);

  const handleStepForward = useCallback(() => {
    if (usedCanonicalRef.current) {
      const retargeted = currentRetargetedRef.current;
      if (retargeted.length === 0) return;
      const next = (effectiveFrameIndex + 1) % retargeted.length;
      prevRetargetedRef.current = retargeted[effectiveFrameIndex];
      targetRetargetedRef.current = retargeted[next];
      transitionProgressRef.current = 0;
      transitionStartTimeRef.current = timeRef.current;
      isAnimatingRef.current = true;
      setFrameIndex(next);
      onFrameChange?.(next, retargeted.length);
      onStepForward?.();
    } else {
      const frames = currentFramesRef.current;
      if (frames.length === 0) return;
      const next = (effectiveFrameIndex + 1) % frames.length;
      prevFrameRef.current = frames[effectiveFrameIndex];
      targetFrameRef2.current = frames[next];
      transitionProgressRef.current = 0;
      transitionStartTimeRef.current = timeRef.current;
      isAnimatingRef.current = true;
      setFrameIndex(next);
      onFrameChange?.(next, frames.length);
      onStepForward?.();
    }
  }, [effectiveFrameIndex, onFrameChange, onStepForward]);

  const handleStepBack = useCallback(() => {
    if (usedCanonicalRef.current) {
      const retargeted = currentRetargetedRef.current;
      if (retargeted.length === 0) return;
      const prev = (effectiveFrameIndex - 1 + retargeted.length) % retargeted.length;
      prevRetargetedRef.current = retargeted[prev];
      targetRetargetedRef.current = retargeted[effectiveFrameIndex];
      transitionProgressRef.current = 0;
      transitionStartTimeRef.current = timeRef.current;
      isAnimatingRef.current = true;
      setFrameIndex(prev);
      onFrameChange?.(prev, retargeted.length);
      onStepBack?.();
    } else {
      const frames = currentFramesRef.current;
      if (frames.length === 0) return;
      const prev = (effectiveFrameIndex - 1 + frames.length) % frames.length;
      prevFrameRef.current = frames[prev];
      targetFrameRef2.current = frames[effectiveFrameIndex];
      transitionProgressRef.current = 0;
      transitionStartTimeRef.current = timeRef.current;
      isAnimatingRef.current = true;
      setFrameIndex(prev);
      onFrameChange?.(prev, frames.length);
      onStepBack?.();
    }
  }, [effectiveFrameIndex, onFrameChange, onStepBack]);

  const handleSpeedChange = useCallback((speed: number) => {
    setPlaybackSpeed(speed);
  }, []);

  return (
    <div className="relative w-full h-96 rounded-xl overflow-hidden border border-slate-700/50">
      <div ref={containerRef} className="w-full h-full" />
      {!modelReady && !loadingError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[#050810]">
          <div className="flex size-12 items-center justify-center rounded-full border-4 border-white/10 border-t-cyan-400 animate-spin" aria-hidden="true" />
          <p className="text-sm font-semibold text-slate-400">Loading 3D avatar…</p>
        </div>
      )}
      {loadingError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-[#050810] p-6 text-center">
          <p className="text-sm font-bold text-red-300">3D Avatar Error</p>
          <p className="text-xs text-slate-500 max-w-xs">{loadingError}</p>
        </div>
      )}
      {totalFrames > 0 && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={handleStepBack}
                className="w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm flex items-center justify-center transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
                title="Step back"
              >
                {"\u25C0"}
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm flex items-center justify-center transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
                title={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? "\u275A\u275A" : "\u25B6"}
              </button>
              <button
                onClick={handleStepForward}
                className="w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm flex items-center justify-center transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
                title="Step forward"
              >
                {"\u25B6\u25B6"}
              </button>
            </div>
            <div className="text-xs text-slate-400 font-mono">
              {effectiveFrameIndex + 1} / {totalFrames}
            </div>
            <div className="flex items-center gap-1">
              {PLAYBACK_SPEEDS.map(speed => (
                <button
                  key={speed}
                  onClick={() => handleSpeedChange(speed)}
                  className={`px-2 py-0.5 rounded text-xs font-bold transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40 ${
                    playbackSpeed === speed
                      ? "bg-cyan-400/30 text-cyan-300 border border-cyan-400/50"
                      : "bg-white/5 text-slate-400 hover:bg-white/10"
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
          <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-cyan-400 rounded-full transition-all duration-100"
              style={{ width: `${((effectiveFrameIndex + 1) / totalFrames) * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
