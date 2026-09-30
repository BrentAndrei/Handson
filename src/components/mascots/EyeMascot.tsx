import { useCallback, useEffect, useRef, useState } from "react";
import { usePointerIntent, prefersReducedMotion } from "../../lib/usePointerIntent";

interface EyeMascotProps {
  className?: string;
  /** Bumps the tracking gain while the card is hovered or focused. */
  attentive?: boolean;
}

/**
 * THE EYE — mascot for the real-time translator card.
 *
 * Concept: SEE = LOOK = RECOGNISE = CAMERA.
 *
 * Construction notes
 * ------------------
 * The eyeball is drawn as an irregular BLOB rather than a symmetrical lens.
 * That single decision is most of the neo-brutalist character: a perfect lens
 * reads as a stock icon, a lopsided one reads as something drawn by hand.
 *
 * The pupil and iris are a separate group translated by --mx/--my. Travel is
 * deliberately small (9px on a ~200px eye) so the pupil never approaches the
 * outline; the clamp inside usePointerIntent guarantees the hard limit.
 *
 * Blink is a scaleY on a lid pair, not a shape swap, so it costs nothing. It
 * fires on a slow irregular cadence and on click, and is disabled entirely
 * under `prefers-reduced-motion`.
 */
export function EyeMascot({ className, attentive = false }: EyeMascotProps) {
  const host = useRef<HTMLDivElement | null>(null);
  const [blink, setBlink] = useState(false);
  const blinkTimer = useRef<number | null>(null);

  usePointerIntent(host, { ease: 0.1 });

  const triggerBlink = useCallback(() => {
    if (prefersReducedMotion()) return;
    setBlink(true);
    if (blinkTimer.current) window.clearTimeout(blinkTimer.current);
    // 120ms closed reads as a blink, not a wink.
    blinkTimer.current = window.setTimeout(() => setBlink(false), 120);
  }, []);

  // Idle cadence: a blink every 3.5-6.5s, deliberately irregular so the eye
  // never settles into a metronome the user starts noticing.
  useEffect(() => {
    if (prefersReducedMotion()) return;
    let alive = true;
    let timeout: number;
    const schedule = () => {
      timeout = window.setTimeout(() => {
        if (!alive) return;
        triggerBlink();
        schedule();
      }, 3500 + Math.random() * 3000);
    };
    schedule();
    return () => {
      alive = false;
      window.clearTimeout(timeout);
      if (blinkTimer.current) window.clearTimeout(blinkTimer.current);
    };
  }, [triggerBlink]);

  const lidScale = blink ? 1 : 0;
  const still = prefersReducedMotion();

  return (
    <div
      ref={host}
      className={className}
      onClick={triggerBlink}
      role="img"
      /* Stable hook for the UI gate, which must distinguish mascots from the
         small sticker icons also present on the card. */
      data-mascot="eye"
      aria-label="An illustrated eye that looks toward your pointer. The real-time translator watches your hands through the camera."
    >
      <svg viewBox="0 0 240 190" className="h-full w-full" aria-hidden focusable="false">
        <defs>
          {/* Same blob as the fill, so the lids can never overshoot the edge. */}
          <clipPath id="hs-eye-clip">
            <path d="M18 96c0-42 46-74 102-74s102 32 102 74-46 74-102 74S18 138 18 96Z" />
          </clipPath>
        </defs>

        {/* Hard offset shadow: the eye sits ON the slab, not inside a screen. */}
        <path
          d="M26 104c0-42 46-74 102-74s102 32 102 74-46 74-102 74S26 146 26 104Z"
          fill="var(--ink)"
        />
        <path
          d="M18 96c0-42 46-74 102-74s102 32 102 74-46 74-102 74S18 138 18 96Z"
          fill="var(--paper-raised)"
          stroke="var(--ink)"
          strokeWidth={7}
        />

        <g clipPath="url(#hs-eye-clip)">
          {/* The tracking group: iris + pupil + highlights move together.
              `--mx`/`--my` are unitless (see usePointerIntent), so multiplying
              into a px length here is valid. 9px on a ~200px eye keeps the
              pupil well clear of the outline. */}
          <g
            style={{
              transform: "translate(calc(var(--mx, 0) * 9px), calc(var(--my, 0) * 9px))",
              transition: still ? undefined : "transform 60ms linear",
            }}
          >
            <circle cx={120} cy={96} r={44} fill="var(--teal)" stroke="var(--ink)" strokeWidth={6} />
            <circle cx={120} cy={96} r={24} fill="var(--ink)" />
            {/* Two hard highlights: the gloss that makes it read as drawn. */}
            <circle cx={106} cy={80} r={9} fill="var(--paper-raised)" />
            <circle cx={134} cy={114} r={5} fill="var(--paper-raised)" />
          </g>

          {/* Lids scale in from the top and bottom edges to close the eye. */}
          <g
            style={{
              transform: `scaleY(${lidScale})`,
              transformOrigin: "120px 22px",
              transition: still ? undefined : "transform 90ms cubic-bezier(0.2,0,0,1)",
            }}
          >
            <rect x={0} y={0} width={240} height={74} fill="var(--ink)" />
          </g>
          <g
            style={{
              transform: `scaleY(${lidScale})`,
              transformOrigin: "120px 170px",
              transition: still ? undefined : "transform 90ms cubic-bezier(0.2,0,0,1)",
            }}
          >
            <rect x={0} y={116} width={240} height={74} fill="var(--ink)" />
          </g>
        </g>

        {/* Re-stroke the outline on top so the lids can never cover it. */}
        <path
          d="M18 96c0-42 46-74 102-74s102 32 102 74-46 74-102 74S18 138 18 96Z"
          fill="none"
          stroke="var(--ink)"
          strokeWidth={7}
        />

        {/* Scan brackets: this is a camera pointed at a hand, not an eye in a
            socket. They also break the silhouette, which stops the mascot from
            reading as a tidy centred icon. */}
        <g stroke="var(--ink)" strokeWidth={6} fill="none">
          <path d="M-6 34V12a8 8 0 0 1 8-8h22" />
          <path d="M246 34V12a8 8 0 0 0-8-8h-22" />
          <path d="M-6 158v22a8 8 0 0 0 8 8h22" />
          <path d="M246 158v22a8 8 0 0 1-8 8h-22" />
        </g>

        {/* Corner tracking points, as if the camera is locking onto landmarks. */}
        <g fill="var(--ink)">
          <rect x={52} y={62} width={7} height={7} />
          <rect x={180} y={58} width={7} height={7} />
          <rect x={58} y={124} width={7} height={7} />
          <rect x={186} y={128} width={7} height={7} />
        </g>

        {/* Attentive state: the tracking points swell and turn coral when the
            card is engaged, so "it noticed you" is legible without motion. */}
        {attentive ? (
          <g fill="var(--coral)" stroke="var(--ink)" strokeWidth={3}>
            <rect x={46} y={56} width={19} height={19} />
            <rect x={174} y={52} width={19} height={19} />
            <rect x={52} y={118} width={19} height={19} />
            <rect x={180} y={122} width={19} height={19} />
          </g>
        ) : null}
      </svg>
    </div>
  );
}
