import { Tooltip } from "./Tooltip";
import { motion } from "framer-motion";

interface Props {
  currentWords: string[];
  currentEnglish: string;
  finalizedSentences: string[];
  finalizedEnglish: string[];
  onClear: () => void;
  onEndSentence: () => void;
  showEnglish: boolean;
  onToggleEnglish: () => void;
}

/**
 * The sentence-building panel — accumulates gloss tokens into a sentence,
 * with End/Clear controls and a recent-sentence history below.
 *
 * The current sentence line is the second piece of hero content: large,
 * high-contrast text on the glass panel so the user can read what they've
 * signed at a glance.
 *
 * `showEnglish` switches the live line between the raw FSL gloss tokens
 * (what the model actually classified) and the rule-based English rendering
 * produced by grammarParser.ts (canonicalisation + contextual
 * disambiguation + Topic-Comment re-ordering + phrase templates). It is NOT
 * a neural translation — there is no parallel corpus to learn one from — so
 * the label says "English sentence" rather than "Translate".
 */
export function SentenceOverlay({
  currentWords,
  currentEnglish,
  finalizedSentences,
  finalizedEnglish,
  onClear,
  onEndSentence,
  showEnglish,
  onToggleEnglish,
}: Props) {
  const listItemVariants = {
    initial: { opacity: 0, y: 8, scale: 0.97 },
    animate: { opacity: 1, y: 0, scale: 1 },
    exit: { opacity: 0, y: -4, scale: 0.97 },
  };
  return (
    <div className="glass-card">
      <motion.div
        className="flex items-center justify-between gap-3"
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
      >
        <Tooltip text="Each committed sign becomes a gloss token. Pause briefly between words; lower your hands to end the sentence.">
          <h3 className="card-title">Sentence mode</h3>
        </Tooltip>
        <label className="flex cursor-pointer items-center gap-2 text-xs font-semibold text-slate-300">
          <input
            type="checkbox"
            checked={showEnglish}
            onChange={onToggleEnglish}
            className="size-4 rounded border-cyan-400/30 bg-white/5 text-cyan-400 focus:ring-2 focus:ring-cyan-400/40"
          />
          <span className="whitespace-nowrap">Show English sentence</span>
        </label>
      </motion.div>
      <motion.p
        className="mt-4 min-h-[2.5rem] hero-text-xl font-bold leading-snug text-white"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.05 }}
      >
        {currentWords.length > 0 ? (
          showEnglish ? (
            currentEnglish || currentWords.join(" ")
          ) : (
            currentWords.join(" ")
          )
        ) : (
          <span className="font-normal text-slate-500">Sign a word to start a sentence…</span>
        )}
      </motion.p>
      <motion.div
        className="mt-4 flex gap-2"
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, delay: 0.1 }}
      >
        <button
          disabled={currentWords.length === 0}
          onClick={onEndSentence}
          className="rounded-lg bg-cyan-500/80 px-3 py-2 text-sm font-bold text-white shadow-sm backdrop-blur-sm transition-all duration-200 active:scale-[0.98] hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-white/5 disabled:text-slate-500 focus:outline-none focus:ring-4 focus:ring-cyan-400/30"
        >
          End sentence
        </button>
        <button
          disabled={currentWords.length === 0}
          onClick={onClear}
          className="rounded-lg border border-cyan-400/20 px-3 py-2 text-sm font-bold text-slate-200 transition-all duration-200 active:scale-[0.98] hover:bg-cyan-400/10 disabled:cursor-not-allowed disabled:text-slate-600 focus:outline-none focus:ring-4 focus:ring-cyan-400/30"
        >
          Clear
        </button>
      </motion.div>
      {finalizedSentences.length > 0 && (
        <motion.div
          className="mt-5 space-y-3 border-t border-white/10 pt-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, delay: 0.15 }}
        >
          <h4 className="text-xs font-bold uppercase tracking-[0.14em] text-slate-400">Recent sentences</h4>
          <ul className="space-y-3">
            {finalizedSentences.map((sentence, i) => (
              <motion.li
                key={i}
                className="rounded-xl border border-white/5 bg-white/5 p-3"
                variants={listItemVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                transition={{ duration: 0.25, delay: i * 0.05 }}
              >
                <div className="text-xs italic text-slate-400">{sentence}</div>
                <div className="mt-1 font-bold text-white">{finalizedEnglish[i] ?? ""}</div>
              </motion.li>
            ))}
          </ul>
        </motion.div>
      )}
    </div>
  );
}
