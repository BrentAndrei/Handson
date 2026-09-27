import { useCallback, useRef, useState } from "react";
import type { Prediction } from "../inference/useSignInference";
import { glossToEnglish } from "./grammarParser";

// Committing a word into a sentence is a stronger claim than just showing
// a live label, so this is intentionally higher than PredictionOverlay's
// 0.6 display threshold.
const COMMIT_CONFIDENCE_THRESHOLD = 0.7;
// Require the same label across this many consecutive predictions before
// committing it — predictions arrive roughly every stride (~250ms at
// 24fps/stride 6), so 2 in a row is ~500ms of holding the sign, enough to
// filter out a single spurious misclassified window without adding much
// perceived lag.
const MIN_STABLE_PREDICTIONS = 2;
// How long hands can be absent from frame before the in-progress sentence
// is considered finished and gets finalized.
const SENTENCE_END_IDLE_MS = 1800;
const MAX_FINALIZED_SENTENCES = 5;

/**
 * Sentence-mode layer on top of the isolated-sign classifier from
 * useSignInference. This is NOT continuous sign language recognition — a
 * true CTC-style sequence model trained on unsegmented sentence video would
 * be needed for that, and no public FSL dataset currently supports training
 * one. What this does instead: watch the existing per-window predictions for
 * a sign held stably above a confidence threshold, commit it as one "word"
 * once seen consistently, and use hands leaving the frame (a natural pause
 * between signs, or lowering hands at the end of a sentence) as the boundary
 * signal for when a sentence is done. Works reasonably well for signing with
 * brief natural pauses between words; will NOT correctly segment truly
 * continuous, co-articulated signing where one sign blends directly into the
 * next with no pause at all.
 *
 * When a sentence is finalised the raw gloss sequence is passed through the
 * FSL grammar layer (grammarParser.ts) which canonicalises spelling, applies
 * contextual disambiguation, and re-orders Topic-Comment glosses into
 * grammatical English. The hook exposes both the raw gloss sentence and the
 * translated English so the UI can show either.
 *
 * Two separate entry points, called from two different places in
 * App.tsx's capture loop:
 *   onPrediction — call only when a genuinely NEW model prediction
 *     arrives (not every frame), drives the word-commit logic.
 *   onActivity — call every captured frame regardless of whether a new
 *     prediction arrived, drives the idle-based sentence-end check. This
 *     has to run independently of onPrediction because once hands leave
 *     frame, useSignInference stops producing new predictions entirely —
 *     if sentence-ending only happened inside onPrediction, it would
 *     never fire during the very silence it's supposed to detect.
 */
export function useSentenceBuilder() {
  const [currentWords, setCurrentWords] = useState<string[]>([]);
  const [currentEnglish, setCurrentEnglish] = useState<string>("");
  const [finalizedSentences, setFinalizedSentences] = useState<string[]>([]);
  const [finalizedEnglish, setFinalizedEnglish] = useState<string[]>([]);

  const currentWordsRef = useRef<string[]>([]);
  const pendingLabelRef = useRef<string | null>(null);
  const pendingStableCountRef = useRef(0);
  const lastCommittedLabelRef = useRef<string | null>(null);
  const lastActivityMsRef = useRef(0);
  const wasHandsPresentRef = useRef(false);

  // Derive the English translation of the in-progress sentence whenever the
  // raw gloss list changes. This is NOT finalised — the grammar layer is
  // deliberately re-run on the live token list so the user sees the
  // translation update as they sign, not only when the sentence ends.
  const refreshCurrentEnglish = useCallback((words: string[]) => {
    if (words.length === 0) {
      setCurrentEnglish("");
      return;
    }
    const { english } = glossToEnglish(words);
    setCurrentEnglish(english);
  }, []);

  const finalizeSentence = useCallback(() => {
    if (currentWordsRef.current.length === 0) return;
    const glossSentence = currentWordsRef.current.join(" ");
    const { english, trace } = glossToEnglish(currentWordsRef.current);
    currentWordsRef.current = [];
    setCurrentWords([]);
    setCurrentEnglish("");
    setFinalizedSentences((prev) => [glossSentence, ...prev].slice(0, MAX_FINALIZED_SENTENCES));
    setFinalizedEnglish((prev) => [english, ...prev].slice(0, MAX_FINALIZED_SENTENCES));
    if (trace.length > 0) {
      console.log("[grammar]", trace.join(" | "));
    }
    lastCommittedLabelRef.current = null;
  }, []);

  const onPrediction = useCallback((prediction: Prediction, timestampMs: number) => {
    if (prediction.confidence < COMMIT_CONFIDENCE_THRESHOLD) {
      pendingLabelRef.current = null;
      pendingStableCountRef.current = 0;
      return;
    }

    if (prediction.label === pendingLabelRef.current) {
      pendingStableCountRef.current += 1;
    } else {
      pendingLabelRef.current = prediction.label;
      pendingStableCountRef.current = 1;
    }

    const isStable = pendingStableCountRef.current >= MIN_STABLE_PREDICTIONS;
    // Guards against one held sign spamming the sentence with repeated
    // commits every time it re-crosses the stability count — a new commit
    // of the SAME label only happens after either a different label was
    // seen in between, or hands left and re-entered frame (see onActivity).
    const isNewWord = prediction.label !== lastCommittedLabelRef.current;

    if (isStable && isNewWord) {
      currentWordsRef.current = [...currentWordsRef.current, prediction.label];
      setCurrentWords(currentWordsRef.current);
      refreshCurrentEnglish(currentWordsRef.current);
      lastCommittedLabelRef.current = prediction.label;
      lastActivityMsRef.current = timestampMs;
    }
  }, [refreshCurrentEnglish]);

  const onActivity = useCallback(
    (handsPresent: boolean, timestampMs: number) => {
      // Hands re-entering frame after being away is a natural "new
      // gesture" boundary — clears the repeat-guard so the same word can
      // be signed again if that's genuinely what happens next.
      if (handsPresent && !wasHandsPresentRef.current) {
        lastCommittedLabelRef.current = null;
      }
      wasHandsPresentRef.current = handsPresent;

      if (handsPresent) {
        lastActivityMsRef.current = timestampMs;
      } else if (
        currentWordsRef.current.length > 0 &&
        timestampMs - lastActivityMsRef.current > SENTENCE_END_IDLE_MS
      ) {
        finalizeSentence();
      }
    },
    [finalizeSentence]
  );

  const clear = useCallback(() => {
    currentWordsRef.current = [];
    setCurrentWords([]);
    setCurrentEnglish("");
    setFinalizedSentences([]);
    setFinalizedEnglish([]);
    pendingLabelRef.current = null;
    pendingStableCountRef.current = 0;
    lastCommittedLabelRef.current = null;
  }, []);

  return {
    currentWords,
    currentEnglish,
    finalizedSentences,
    finalizedEnglish,
    onPrediction,
    onActivity,
    clear,
    endSentenceNow: finalizeSentence,
  };
}
