import { useEffect, useRef, useState } from "react";
import { usePointerIntent, prefersReducedMotion } from "../../lib/usePointerIntent";

interface GuideMascotProps {
  className?: string;
  /** Rouses the mascot: it straightens up and the spark flares. */
  roused?: boolean;
}

/**
 * THE GUIDE — mascot for the onboarding tour.
 *
 * Concept: SPARK = RECOGNITION = THE MOMENT A SIGN LANDS. Where the
 * eye reads and the hands perform, the guide is the little print
 * character that introduces the two.
 *
 * Construction notes
 * ------------------
 * A blob body (like the eye's, deliberately lopsided so it reads as
 * drawn rather than traced), ONE oversized teal eye, a four-point
 * sun-coloured spark on an antenna, and two coral feet. The feet are
 * separate blocks on purpose: a mascot that stands on something has
 * weight, which is the whole printed-object illusion.
 *
 * `data-mascot="guide"` is a DELIBERATE value. The UI gate's global
 * mascot scan queries [data-mascot="eye"] and [data-mascot="hands"],
 * so a third value keeps this character invisible to those checks
 * wherever the tour renders.
 */
export function GuideMascot({ className, roused = false }: GuideMascotProps) {
  const host = useRef<HTMLDivElement | null>(null);
  const [blink, setBlink] = useState(false);
  const blinkTimer = useRef<number | null>(null);

  usePointerIntent(host, { ease: 0.11 });

  const triggerBlink = useRef(() => {});
  triggerBlink.current = () => {
    if (prefersReducedMotion()) return;
    setBlink(true);
    if (blinkTimer.current) window.clearTimeout(blinkTimer.current);
    blinkTimer.current = window.setTimeout(() => setBlink(false), 130);
  };

  // Idle cadence, offset from the eye's 3.5-6.5s so a tour open on
  // the home screen never blinks in sync with the card mascots.
  useEffect(() => {
    if (prefersReducedMotion()) return;
    let alive = true;
    let timeout: number;
    const schedule = () => {
      timeout = window.setTimeout(() => {
        if (!alive) return;
        triggerBlink.current();
        schedule();
      }, 4200 + Math.random() * 2800);
    };
    schedule();
    return () => {
      alive = false;
      window.clearTimeout(timeout);
      if (blinkTimer.current) window.clearTimeout(blinkTimer.current);
    };
  }, []);

  const lidScale = blink ? 1 : 0;
  const still = prefersReducedMotion();

  return (
    <div
      ref={host}
      className={className}
      role="img"
      data-mascot="guide"
      aria-label="A small printed character with one large eye and a spark on its antenna. It introduces the tour."
    >
      <svg viewBox="0 0 200 190" className="h-full w-full" aria-hidden focusable="false">
        {/* Pointer intent: a few px of drift and a couple of degrees of
            lean. --mx/--my are unitless (see usePointerIntent), so the
            calc() below mixes px and deg legally. */}
        <g
          style={{
            transform: `translate(calc(var(--mx, 0) * 8px), calc(var(--my, 0) * 6px)) rotate(calc(var(--mx, 0) * 4deg))`,
            transition: still ? undefined : "transform 160ms linear",
          }}
        >
          {/* Idle bob. The keyframes live in index.css so the global
              prefers-reduced-motion rule (which kills every CSS
              animation) disables them for free. */}
          <g
            style={{
              animation: still ? undefined : "hs-bob 2.6s ease-in-out infinite alternate",
            }}
          >
            {/* Contact shadow: the character sits ON the surface. */}
            <ellipse cx={100} cy={172} rx={52} ry={9} fill="var(--ink)" />

            {/* Feet — coral blocks, not part of the body outline. */}
            <rect x={66} y={152} width={26} height={17} rx={8} fill="var(--coral)" stroke="var(--ink)" strokeWidth={5} />
            <rect x={108} y={152} width={26} height={17} rx={8} fill="var(--coral)" stroke="var(--ink)" strokeWidth={5} />

            {/* Body: an irregular blob. Asymmetry is the point — a
                perfectly round body reads as a stock icon. */}
            <path
              d="M100 30 C 57 30 32 60 33 100 C 34 141 65 166 102 164 C 141 162 168 136 167 97 C 166 57 143 30 100 30 Z"
              fill="var(--ink)"
              transform="translate(6, 7)"
            />
            <path
              d="M100 30 C 57 30 32 60 33 100 C 34 141 65 166 102 164 C 141 162 168 136 167 97 C 166 57 143 30 100 30 Z"
              fill="var(--paper-raised)"
              stroke="var(--ink)"
              strokeWidth={7}
            />

            {/* Antenna: a stem and a four-point spark, sun-coloured. */}
            <path d="M100 30 C 99 22 99 16 100 10" fill="none" stroke="var(--ink)" strokeWidth={5} strokeLinecap="round" />
            <g
              style={{
                transform: roused && !still ? "translate(0px, -3px)" : undefined,
                transition: still ? undefined : "transform 200ms var(--ease-snap, cubic-bezier(0.2,0,0,1))",
              }}
            >
              <path
                d="M100 -2 L105 5 L112 10 L105 15 L100 22 L95 15 L88 10 L95 5 Z"
                fill="var(--sun)"
                stroke="var(--ink)"
                strokeWidth={4}
                strokeLinejoin="round"
              />
            </g>

            {/* The eye — the largest element, and the character. */}
            <g
              style={{
                transform: `translate(calc(var(--mx, 0) * 5px), calc(var(--my, 0) * 4px))`,
                transition: still ? undefined : "transform 60ms linear",
              }}
            >
              <circle cx={100} cy={90} r={31} fill="var(--teal)" stroke="var(--ink)" strokeWidth={6} />
              <circle cx={100} cy={90} r={14} fill="var(--ink)" />
              <circle cx={91} cy={81} r={6} fill="var(--paper-raised)" />
            </g>

            {/* Blink lid: sweeps down over the eye. Drawn AFTER the eye
                so it can cover it; collapsed to its top edge at rest,
                so it is invisible while open. */}
            <g
              style={{
                transform: `scaleY(${lidScale})`,
                transformOrigin: "100px 50px",
                transition: still ? undefined : "transform 90ms cubic-bezier(0.2,0,0,1)",
              }}
            >
              <rect x={40} y={50} width={120} height={74} fill="var(--paper-raised)" stroke="var(--ink)" strokeWidth={4} />
            </g>

            {/* Mouth: a small ink smile, off-centre like the blob. */}
            <path
              d="M86 132 Q 99 142 114 130"
              fill="none"
              stroke="var(--ink)"
              strokeWidth={5}
              strokeLinecap="round"
            />

            {/* Cheeks — the only decorative colour on the body. */}
            <circle cx={66} cy={116} r={7} fill="var(--pink)" opacity={0.85} />
            <circle cx={134} cy={116} r={7} fill="var(--pink)" opacity={0.85} />
          </g>
        </g>
      </svg>
    </div>
  );
}
