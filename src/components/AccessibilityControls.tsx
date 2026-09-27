interface Props {
  textSize: "sm" | "base" | "lg";
  onTextSize: (size: "sm" | "base" | "lg") => void;
  highContrast: boolean;
  onHighContrast: (on: boolean) => void;
}

/**
 * An unobtrusive accessibility settings cluster.
 *
 * HandsOn's primary users are deaf and hard-of-hearing signers, so readable
 * text and strong contrast matter more than a decorative toggle. These
 * controls live at the bottom-right of the main content area, out of the
 * way of the camera feed and prediction hero text, but reachable without
 * hunting. They scroll with the page to guarantee they never overlap
 * other content.
 *
 * Both toggles are real state, not just visual chrome: `textSize` rescales
 * the whole app via a CSS attribute selector, and `highContrast` swaps the
 * glass panels to opaque dark surfaces with white text.
 */
export function AccessibilityControls({ textSize, onTextSize, highContrast, onHighContrast }: Props) {
  return (
    <div className="flex flex-col items-start gap-2">
      <div className="glass-sm flex items-center gap-1 p-1">
        <span className="px-2 text-[10px] font-bold uppercase tracking-[0.12em] text-slate-400">Aa</span>
        <button
          onClick={() => onTextSize("sm")}
          className={`rounded-md px-2 py-1 text-xs font-bold transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40 ${
            textSize === "sm" ? "bg-cyan-500/30 text-white" : "text-slate-300 hover:bg-white/5"
          }`}
          aria-pressed={textSize === "sm"}
        >
          A
        </button>
        <button
          onClick={() => onTextSize("base")}
          className={`rounded-md px-2 py-1 text-xs font-bold transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40 ${
            textSize === "base" ? "bg-cyan-500/30 text-white" : "text-slate-300 hover:bg-white/5"
          }`}
          aria-pressed={textSize === "base"}
        >
          Aa
        </button>
        <button
          onClick={() => onTextSize("lg")}
          className={`rounded-md px-2 py-1 text-xs font-bold transition-all duration-200 active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-cyan-400/40 ${
            textSize === "lg" ? "bg-cyan-500/30 text-white" : "text-slate-300 hover:bg-white/5"
          }`}
          aria-pressed={textSize === "lg"}
        >
          Aaa
        </button>
      </div>
      <button
        onClick={() => onHighContrast(!highContrast)}
        className="glass-card flex items-center gap-2 px-3 py-1.5 text-xs font-bold text-slate-200 transition-all duration-200 active:scale-[0.98] hover:bg-white/5 focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
        aria-pressed={highContrast}
      >
        <span className="size-2.5 rounded-full bg-current" aria-hidden="true" />
        {highContrast ? "High contrast on" : "High contrast"}
      </button>
    </div>
  );
}
