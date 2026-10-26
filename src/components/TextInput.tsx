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
 * FSL gloss sequence. Styled with hard ink borders on warm paper
 * and teal accents.
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
          className="min-h-[4.5rem] w-full flex-1 resize-y rounded-none border-2 border-ink bg-paper px-4 py-3 text-base text-ink outline-none transition placeholder:text-ink-mute focus:ring-2 focus:ring-teal focus:border-teal disabled:cursor-not-allowed disabled:opacity-50"
        />
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-ink-mute">Enter to submit · Shift+Enter for newline</span>
          <button
            onClick={onSubmit}
            disabled={disabled || value.trim().length === 0}
            className="rounded-none border-2 border-ink bg-ink px-5 py-2.5 font-bold text-paper shadow-brutal-sm transition-all duration-200 active:scale-[0.98] hover:bg-teal hover:border-teal disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-teal"
          >
            Sign this
          </button>
        </div>
      </div>
    </section>
  );
}
