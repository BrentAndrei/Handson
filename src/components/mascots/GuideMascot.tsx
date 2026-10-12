import { useEffect, useRef, useState } from "react";
import { usePointerIntent, prefersReducedMotion } from "../../lib/usePointerIntent";

interface GuideMascotProps {
  className?: string;
  /** Rouses the mascot: eyebrows rise, the spark flares. */
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
 * drawn rather than traced), ONE oversized teal eye that tracks the
 * pointer, ink eyebrows that rise when it is roused, an open smile,
 * coral feet, a sun four-point spark on an antenna — and a waving
 * arm, because a character that greets you is a character.
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
      aria-label="A small printed character with one large eye, a waving arm and a spark on its antenna. It introduces the tour."
    >
      <svg viewBox="0 0 220 200" className="h-full w-full" aria-hidden focusable="false">
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
            {/* Waving arm, drawn FIRST so the shoulder tucks behind
                the body. The wave is a slow rotate about the
                shoulder — a greeting, not a seizure. */}
            <g
              style={{
                transformOrigin: "166px 94px",
                animation: still ? undefined : "hs-wave 1.7s ease-in-out infinite",
              }}
            >
              <rect x={158} y={87} width={36} height={14} rx={7} fill="var(--lime)" stroke="var(--ink)" strokeWidth={4.5} />
              <circle cx={197} cy={94} r={12} fill="var(--lime)" stroke="var(--ink)" strokeWidth={4.5} />
            </g>

            {/* Contact shadow: the character sits ON the surface. */}
            <ellipse cx={110} cy={182} rx={58} ry={9} fill="var(--ink)" />

            {/* Feet — coral blocks, not part of the body outline. */}
            <rect x={74} y={160} width={27} height={18} rx={8} fill="var(--coral)" stroke="var(--ink)" strokeWidth={5} />
            <rect x={119} y={160} width={27} height={18} rx={8} fill="var(--coral)" stroke="var(--ink)" strokeWidth={5} />

            {/* Body: an irregular blob. Asymmetry is the point — a
                perfectly round body reads as a stock icon. */}
            <path
              d="M110 36 C 64 36 38 68 39 108 C 40 150 72 174 111 172 C 152 170 182 142 181 102 C 180 60 156 36 110 36 Z"
              fill="var(--ink)"
              transform="translate(6, 7)"
            />
            <path
              d="M110 36 C 64 36 38 68 39 108 C 40 150 72 174 111 172 C 152 170 182 142 181 102 C 180 60 156 36 110 36 Z"
              fill="var(--paper-raised)"
              stroke="var(--ink)"
              strokeWidth={7}
            />

            {/* Antenna: a stem and a four-point spark, sun-coloured. */}
            <path d="M110 36 C 109 28 109 22 110 18" fill="none" stroke="var(--ink)" strokeWidth={5} strokeLinecap="round" />
            <g
              style={{
                transform: roused && !still ? "translate(0px, -3px)" : undefined,
                transition: still ? undefined : "transform 200ms var(--ease-snap, cubic-bezier(0.2,0,0,1))",
              }}
            >
              <path
                d="M110 0 L115 7 L122 12 L115 17 L110 24 L105 17 L98 12 L105 7 Z"
                fill="var(--sun)"
                stroke="var(--ink)"
                strokeWidth={4}
                strokeLinejoin="round"
              />
            </g>

            {/* Eyebrows — the expression. They rise when roused. */}
            <g
              style={{
                transform: roused && !still ? "translate(0px, -4px)" : undefined,
                transition: still ? undefined : "transform 180ms var(--ease-snap, cubic-bezier(0.2,0,0,1))",
              }}
            >
              <rect x={76} y={50} width={27} height={7} rx={3.5} fill="var(--ink)" />
              <rect x={117} y={50} width={27} height={7} rx={3.5} fill="var(--ink)" />
            </g>

            {/* The eye — the largest element, and the character. */}
            <g
              style={{
                transform: `translate(calc(var(--mx, 0) * 5px), calc(var(--my, 0) * 4px))`,
                transition: still ? undefined : "transform 60ms linear",
              }}
            >
              <circle cx={110} cy={98} r={34} fill="var(--teal)" stroke="var(--ink)" strokeWidth={6} />
              <circle cx={110} cy={98} r={15} fill="var(--ink)" />
              <circle cx={99} cy={87} r={6.5} fill="var(--paper-raised)" />
              <circle cx={121} cy={108} r={4} fill="var(--paper-raised)" />
            </g>

            {/* Blink lid: sweeps down over the eye. Drawn AFTER the eye
                so it can cover it; collapsed to its top edge at rest,
                so it is invisible while open. */}
            <g
              style={{
                transform: `scaleY(${lidScale})`,
                transformOrigin: "110px 60px",
                transition: still ? undefined : "transform 90ms cubic-bezier(0.2,0,0,1)",
              }}
            >
              <rect x={64} y={60} width={92} height={76} fill="var(--paper-raised)" stroke="var(--ink)" strokeWidth={4} />
            </g>

            {/* Open smile — a half-disc, the friendliest geometry. */}
            <path d="M88 128 A22 17 0 0 1 132 128 Z" fill="var(--ink)" />

            {/* Cheeks — the only decorative colour on the body. */}
            <circle cx={70} cy={122} r={7} fill="var(--pink)" opacity={0.85} />
            <circle cx={150} cy={122} r={7} fill="var(--pink)" opacity={0.85} />
          </g>
        </g>
      </svg>
    </div>
  );
}
