interface Props {
  text: string;
  children: React.ReactNode;
  side?: "top" | "bottom";
}

/**
 * Wraps a label (or any element) with an inline info icon that reveals a
 * tooltip on hover or keyboard focus. Renders as "Label i".
 *
 * The tooltip is hidden by default and shown on hover/focus of the icon, so
 * it never blocks interaction and is fully keyboard-accessible. Text is
 * short plain-English: this is a field tool for teachers, not a spec doc.
 */
export function Tooltip({ text, children, side = "bottom" }: Props) {
  return (
    <span className="tooltip-group inline-flex items-center">
      {children}
      <span
        className="tooltip-icon"
        aria-label={text}
        tabIndex={0}
      >i</span>
      <span
        role="tooltip"
        className={`tooltip-pop invisible opacity-0 transition-opacity duration-150 hover:visible hover:opacity-100 focus-within:visible focus-within:opacity-100 ${side === "top" ? "bottom-[calc(100%+0.5rem)]" : ""}`}
      >
        {text}
      </span>
    </span>
  );
}
