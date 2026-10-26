import type { Prediction } from "../inference/useSignInference";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  ready: boolean;
  error: string | null;
  prediction: Prediction | null;
  confidenceThreshold?: number;
  secondOpinion?: { consulted: boolean; switched: boolean } | null;
  onThresholdChange?: (threshold: number) => void;
}

/**
 * The live single-word prediction — the hero content of Predict mode.
 *
 * The predicted gloss sits in large, high-contrast ink text on warm
 * paper so it reads as the point of the app, not decoration. A
 * confidence bar below it gives a quick visual cue: lime when the
 * model is confident, sun when it wants a clearer sign.
 */
export function PredictionOverlay({
  ready,
  error,
  prediction,
  confidenceThreshold = 0.6,
  secondOpinion = null,
  onThresholdChange,
}: Props) {
  const confident = !!prediction && prediction.confidence >= confidenceThreshold;
  const fadeIn = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: -4 } };
  const fadeInSlow = { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 } };
  return (
    <section aria-live="polite" className="glass-card">
      <motion.p {...fadeIn} className="card-title">Live prediction</motion.p>
      <AnimatePresence mode="wait">
        {error && (
          <motion.p
            className="mt-4 rounded-none border-2 border-coral bg-coral/10 p-3 text-coral"
            {...fadeIn}
            key="error"
          >
            Model error: {error}
          </motion.p>
        )}
        {!error && !ready && (
          <motion.div
            className="mt-5 flex flex-col items-center gap-5 py-6"
            {...fadeInSlow}
            key="loading"
            transition={{ duration: 0.4 }}
          >
            <motion.div
              className="flex size-14 items-center justify-center rounded-full border-4 border-ink/15 border-t-teal animate-spin"
              aria-hidden="true"
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
            />
            <div className="text-center">
              <p className="font-semibold text-ink">Loading FSL model</p>
              <p className="mt-1 text-sm text-ink-soft">Downloading the BiGRU model and warm-up pass — this takes a few moments.</p>
            </div>
          </motion.div>
        )}
        {!error && ready && !prediction && (
          <motion.div
            className="mt-5 flex flex-col items-center gap-4 py-6 text-center"
            {...fadeIn}
            key="ready"
            transition={{ duration: 0.35 }}
          >
            <div className="flex size-14 items-center justify-center rounded-full border-2 border-teal bg-teal/10 text-teal-deep">
              <svg className="size-7" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M12 2a3 3 0 0 0-3 3v1a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
                <path d="M19 12a7 7 0 0 1-14 0" />
                <circle cx="8" cy="17" r="2" />
                <circle cx="16" cy="17" r="2" />
              </svg>
            </div>
            <div>
              <p className="font-semibold text-ink">Model ready</p>
              <p className="mt-1 text-sm text-ink-soft">Perform a sign to get a prediction.</p>
            </div>
          </motion.div>
        )}
        {!error && ready && prediction && (
          <motion.div
            className="mt-5"
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -4 }}
            transition={{ duration: 0.35, ease: "easeOut" }}
            key="prediction"
          >
            <div className="flex flex-col items-center gap-4 text-center">
              <motion.p
                className={`m-0 break-words hero-text font-black tracking-tight ${
                  confident ? "text-teal-deep" : "text-sun"
                }`}
                transition={{ duration: 0.6 }}
              >
                {prediction.label}
              </motion.p>
              <motion.span
                className={`pill ${
                  confident ? "bg-teal text-ink" : "bg-sun text-ink"
                }`}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.25 }}
              >
                {confident ? "Confident match" : "Keep signing for a clearer match"}
              </motion.span>
            </div>
            <motion.div
              className="mt-5"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.15, duration: 0.3 }}
            >
              <div className="flex items-center justify-between text-sm font-semibold text-ink-soft">
                <span>Confidence</span>
                <span>{(prediction.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mt-2 h-3 overflow-hidden rounded-full bg-ink/10">
                <motion.div
                  className={`h-full rounded-full ${
                    confident ? "bg-blue-500" : "bg-sun"
                  }`}
                  initial={{ width: "0%" }}
                  animate={{ width: `${Math.round(prediction.confidence * 100)}%` }}
                  transition={{ delay: 0.2, duration: 0.5, ease: "easeOut" }}
                />
              </div>
            </motion.div>
            {/* Confidence threshold slider */}
            <div className="mt-3 flex items-center gap-2 text-xs">
              <button
                onClick={() => {
                  const next = Math.max(0.1, confidenceThreshold - 0.1);
                  onThresholdChange?.(next);
                }}
                className="px-2 py-0.5 rounded bg-ink/5 hover:bg-ink/10 transition-colors"
              >−</button>
              <input
                type="range"
                min={10}
                max={95}
                step={5}
                value={Math.round(confidenceThreshold * 100)}
                onChange={(e) => {
                  const next = Number(e.target.value) / 100;
                  onThresholdChange?.(next);
                }}
                className="w-24 accent-[var(--ink)]"
              />
              <button
                onClick={() => {
                  const next = Math.min(0.95, confidenceThreshold + 0.1);
                  onThresholdChange?.(next);
                }}
                className="px-2 py-0.5 rounded bg-ink/5 hover:bg-ink/10 transition-colors"
              >+</button>
              <span className="ml-auto w-12 text-center font-semibold text-ink">
                {(confidenceThreshold * 100).toFixed(0)}%
              </span>
            </div>
            {/* Second opinion consult indicator */}
            {secondOpinion !== null && (
              <motion.p
                className="mt-2 flex items-center gap-1.5 text-xs font-medium text-sun"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.2 }}
              >
                <span className="inline-block size-1.5 rounded-full bg-sun" aria-hidden="true" />
                Second opinion consult...
              </motion.p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}