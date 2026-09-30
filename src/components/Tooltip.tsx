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
      {/* `side` is applied as a modifier class. The popover's own position
          defaults to above the trigger; `tooltip-below` flips it. */}
      <span role="tooltip" className={`tooltip-pop ${side === "top" ? "tooltip-above" : "tooltip-below"}`}>
        {text}
      </span>
    </span>
  );
}
