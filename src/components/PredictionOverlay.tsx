import type { Prediction } from "../inference/useSignInference";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  ready: boolean;
  error: string | null;
  prediction: Prediction | null;
  confidenceThreshold?: number;
}

/**
 * The live single-word prediction — the hero content of Predict mode.
 *
 * The predicted gloss sits in large, high-contrast text inside a frosted
 * glass panel so it reads as the point of the app, not decoration. A
 * confidence bar below it gives a quick visual cue: green when the model
 * is confident, amber when it wants a clearer sign.
 */
export function PredictionOverlay({ ready, error, prediction, confidenceThreshold = 0.6 }: Props) {
  const confident = !!prediction && prediction.confidence >= confidenceThreshold;
  const fadeIn = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: -4 } };
  const fadeInSlow = { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 } };
  return (
    <section aria-live="polite" className="glass-card">
      <motion.p {...fadeIn} className="card-title">Live prediction</motion.p>
      <AnimatePresence mode="wait">
        {error && (
          <motion.p
            className="mt-4 rounded-xl border border-red-400/20 bg-red-400/10 p-3 text-red-200"
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
              className="flex size-14 items-center justify-center rounded-full border-4 border-white/15 border-t-cyan-400 animate-spin"
              aria-hidden="true"
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
            />
            <div className="text-center">
              <p className="font-semibold text-slate-200">Loading FSL model</p>
              <p className="mt-1 text-sm text-slate-400">Downloading the BiGRU model and warm-up pass — this takes a few moments.</p>
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
            <div className="flex size-14 items-center justify-center rounded-full border border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
              <svg className="size-7" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M12 2a3 3 0 0 0-3 3v1a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
                <path d="M19 12a7 7 0 0 1-14 0" />
                <circle cx="8" cy="17" r="2" />
                <circle cx="16" cy="17" r="2" />
              </svg>
            </div>
            <div>
              <p className="font-semibold text-slate-200">Model ready</p>
              <p className="mt-1 text-sm text-slate-400">Perform a sign to get a prediction.</p>
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
                  confident ? "text-cyan-300" : "text-amber-300"
                }`}
                animate={{ textShadow: confident ? "0 0 30px rgba(6,182,212,0.5)" : "0 0 30px rgba(245,158,11,0.4)" }}
                transition={{ duration: 0.6 }}
              >
                {prediction.label}
              </motion.p>
              <motion.span
                className={`pill ${
                  confident ? "bg-cyan-400/15 text-cyan-200" : "bg-amber-400/15 text-amber-200"
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
              <div className="flex items-center justify-between text-sm font-semibold text-slate-300">
                <span>Confidence</span>
                <span>{(prediction.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mt-2 h-3 overflow-hidden rounded-full bg-white/10">
                <motion.div
                  className={`h-full rounded-full ${
                    confident ? "bg-cyan-400" : "bg-amber-400"
                  }`}
                  initial={{ width: "0%" }}
                  animate={{ width: `${Math.round(prediction.confidence * 100)}%` }}
                  transition={{ delay: 0.2, duration: 0.5, ease: "easeOut" }}
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}