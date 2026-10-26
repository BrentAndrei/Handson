import { useState } from "react";
import type { EnglishToGlossResult } from "../features/englishToGloss";

interface Props {
  result: EnglishToGlossResult;
}

/**
 * Displays a structured FSL gloss sequence produced by
 * englishToGloss.ts.
 *
 * Each gloss token shows its FSL label and English meaning.
 * Topic/Comment roles are labelled so a hearing user can see
 * the Topic-Comment structure. Unknown words are flagged.
 *
 * A "Show trace" toggle reveals the audit log of every
 * transformation applied, so a teacher can see exactly
 * why a given English sentence produced a given gloss order.
 */
export function GlossDisplay({ result }: Props) {
  const { glosses, sentenceType, trace } = result;
  const [showTrace, setShowTrace] = useState(false);
  if (glosses.length === 0) return null;

  return (
    <section className="glass-card">
      <div className="flex items-center justify-between gap-3">
        <p className="card-title">FSL Gloss Sequence</p>
        <div className="flex items-center gap-2">
          {trace.length > 0 && (
            <button
              onClick={() => setShowTrace((v) => !v)}
              className="rounded-none border-2 border-ink px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.12em] text-ink-soft transition-all duration-200 active:scale-[0.98] hover:bg-ink hover:text-paper focus:outline-none focus:ring-2 focus:ring-teal"
              aria-expanded={showTrace}
            >
              {showTrace ? "Hide trace" : "Trace"}
            </button>
          )}
          <span className="pill bg-teal text-ink">{sentenceType}</span>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {glosses.map((g, i) => (
          <div
            key={i}
            className={`rounded-none border-2 px-3 py-2 ${
              g.role === "topic"
                ? "border-teal bg-teal/5"
                : g.role === "comment"
                ? "border-ink bg-paper-raised"
                : "border-ink/30 bg-paper"
            }`}
          >
            <p className={`m-0 text-xs font-bold uppercase tracking-[0.12em] ${
              g.role === "topic" ? "text-teal-deep" : "text-ink-mute"
            }`}>
              {g.gloss}
            </p>
            <p className="mt-0.5 text-sm font-semibold text-ink">{g.meaning}</p>
            <p className={`m-0 text-[10px] font-bold uppercase tracking-[0.1em] ${
              g.role === "topic" ? "text-teal-deep" : "text-ink-mute"
            }`}>
              {g.role}
            </p>
          </div>
        ))}
      </div>
      {showTrace && trace.length > 0 && (
        <div className="mt-4 max-h-40 overflow-auto rounded-none border-2 border-ink bg-paper p-3">
          <ul className="m-0 space-y-1 list-inside text-xs text-ink-soft">
            {trace.map((t, i) => (
              <li key={i} className="font-mono break-words">{t}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
