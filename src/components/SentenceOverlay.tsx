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
 * high-contrast ink text on warm paper so the user can read what they've
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
        <label className="flex cursor-pointer items-center gap-2 text-xs font-semibold text-ink-soft">
          <input
            type="checkbox"
            checked={showEnglish}
            onChange={onToggleEnglish}
            className="size-4 rounded-none border-2 border-ink bg-paper text-teal focus:ring-2 focus:ring-teal"
          />
          <span className="whitespace-nowrap">Show English sentence</span>
        </label>
      </motion.div>
      <motion.p
        className="mt-4 min-h-[2.5rem] hero-text-xl font-bold leading-snug text-ink"
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
          <span className="font-normal text-ink-mute">Sign a word to start a sentence…</span>
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
          className="rounded-none border-2 border-ink bg-ink px-3 py-2 text-sm font-bold text-paper shadow-brutal-sm transition-all duration-200 active:scale-[0.98] hover:bg-teal hover:border-teal disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-teal"
        >
          End sentence
        </button>
        <button
          disabled={currentWords.length === 0}
          onClick={onClear}
          className="rounded-none border-2 border-ink bg-paper px-3 py-2 text-sm font-bold text-ink shadow-brutal-sm transition-all duration-200 active:scale-[0.98] hover:bg-ink hover:text-paper disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-teal"
        >
          Clear
        </button>
      </motion.div>
      {finalizedSentences.length > 0 && (
        <motion.div
          className="mt-5 space-y-3 border-t-2 border-ink pt-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, delay: 0.15 }}
        >
          <h4 className="text-xs font-bold uppercase tracking-[0.14em] text-ink-mute">Recent sentences</h4>
          <ul className="space-y-3">
            {finalizedSentences.map((sentence, i) => (
              <motion.li
                key={i}
                className="rounded-none border-2 border-ink bg-paper p-3"
                variants={listItemVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                transition={{ duration: 0.25, delay: i * 0.05 }}
              >
                <div className="text-xs italic text-ink-mute">{sentence}</div>
                <div className="mt-1 font-bold text-ink">{finalizedEnglish[i] ?? ""}</div>
              </motion.li>
            ))}
          </ul>
        </motion.div>
      )}
    </div>
  );
}
