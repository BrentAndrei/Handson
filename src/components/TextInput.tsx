interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

/**
 * Typed English text input for the Text-to-Sign pipeline.
 *
 * A hearing user types English and the app produces a structured
 * FSL gloss sequence. Styled to match the glass aesthetic with
 * cyan accents.
 */
export function TextInput({ value, onChange, onSubmit, disabled }: Props) {
  return (
    <section className="glass-card">
      <p className="card-title">Type English</p>
      <div className="mt-4 flex flex-col gap-3">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !disabled && value.trim().length > 0) {
              e.preventDefault();
              onSubmit();
            }
          }}
          placeholder="Type a sentence in English, e.g. How are you? (Enter to sign)"
          disabled={disabled}
          rows={3}
          className="min-h-[4.5rem] w-full flex-1 resize-y rounded-xl border border-cyan-400/20 bg-white/5 px-4 py-3 text-base text-white outline-none transition placeholder:text-slate-500 focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
        />
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-500">Enter to submit · Shift+Enter for newline</span>
          <button
            onClick={onSubmit}
            disabled={disabled || value.trim().length === 0}
            className="rounded-xl bg-cyan-500/80 px-5 py-2.5 font-bold text-white shadow-sm backdrop-blur-sm transition-all duration-200 active:scale-[0.98] hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-white/5 disabled:text-slate-500 focus:outline-none focus:ring-4 focus:ring-cyan-400/30"
          >
            Sign this
          </button>
        </div>
      </div>
    </section>
  );
}
