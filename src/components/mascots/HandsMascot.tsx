import { useEffect, useRef, useState } from "react";
import { usePointerIntent, prefersReducedMotion } from "../../lib/usePointerIntent";

interface HandsMascotProps {
  className?: string;
  /** Rises the hands and adds the motion cues when the card is engaged. */
  active?: boolean;
}

/**
 * THE HANDS â€” mascot for the 3D sign avatar card.
 *
 * Concept: SIGN = GESTURE = COMMUNICATION = 3D REPRESENTATION.
 *
 * These are deliberately NOT emoji and NOT a stock hand icon. Each hand is a
 * single closed path: a squared palm, a thumb block, three folded-finger
 * notches, and one tall index finger. The fold notches are what make it read
 * as a hand rather than as a mitten.
 *
 * ASYMMETRY IS THE POINT. The two hands are not mirror images: the left is
 * rotated -3deg, sits 8px higher, and uses lime; the right is +4deg, sits
 * lower, and uses sun. Perfect symmetry here is what makes mascot illustrations
 * look like clip art.
 *
 * Follow behaviour uses --mx/--my translated into a small rotation plus a few
 * px of lean. Rotation is capped hard (see ROT_X / ROT_Y) because the brief
 * is "noticing you", not "chasing you".
 */
const ROT_X = 7;   // max degrees of lean left/right
const PX_SOFT = 14; // max px of drift across the composition
const PX_Y = 10;    // max px of vertical drift

/**
 * ONE HAND, BUILT FROM PRIMITIVES RATHER THAN A SINGLE OUTLINE PATH.
 *
 * WHY NOT A PATH: an anatomically plausible hand outline collapses into an
 * unreadable blob at 120px, which is exactly what the first attempt did. Neo-
 * brutalist illustration is GEOMETRIC anyway, so each finger is a rounded
 * rectangle with a thick ink stroke. The hand stays legible at any size because
 * there is no fine detail to lose, and the construction is honest about being
 * drawn rather than traced.
 *
 * Anatomy: a palm block, one tall index finger, three shorter folded fingers
 * reading right-to-left, and a thumb block on the left.
 */
function Hand({ fill, shadow = true }: { fill: string; shadow?: boolean }) {
  const S = 6; // stroke width
  const outline = "var(--ink)";
  return (
    <g>
      {shadow ? (
        <g transform="translate(8, 8)" opacity="1">
          <rect x="26" y="62" width="76" height="82" rx="10" fill="var(--ink)" />
          <rect x="44" y="6" width="24" height="62" rx="12" fill="var(--ink)" />
          <rect x="4" y="76" width="30" height="26" rx="12" fill="var(--ink)" />
          <rect x="66" y="54" width="42" height="24" rx="11" fill="var(--ink)" />
          <rect x="66" y="80" width="42" height="24" rx="11" fill="var(--ink)" />
          <rect x="66" y="106" width="40" height="24" rx="11" fill="var(--ink)" />
        </g>
      ) : null}

      {/* Index finger â€” the tallest element, and the whole point of the glyph. */}
      <rect x="44" y="6" width="24" height="64" rx="12" fill={fill} stroke={outline} strokeWidth={S} />
      {/* Palm */}
      <rect x="26" y="62" width="76" height="82" rx="10" fill={fill} stroke={outline} strokeWidth={S} />
      {/* Three folded fingers, stepping down to the right. */}
      <rect x="66" y="54" width="42" height="26" rx="12" fill={fill} stroke={outline} strokeWidth={S} />
      <rect x="66" y="80" width="42" height="26" rx="12" fill={fill} stroke={outline} strokeWidth={S} />
      <rect x="66" y="106" width="40" height="26" rx="12" fill={fill} stroke={outline} strokeWidth={S} />
      {/* Thumb, angled off the left of the palm. */}
      <rect x="4" y="76" width="32" height="27" rx="12" fill={fill} stroke={outline} strokeWidth={S} />
    </g>
  );
}

export function HandsMascot({ className, active = false }: HandsMascotProps) {
  const host = useRef<HTMLDivElement | null>(null);
  const [lifted, setLifted] = useState(false);

  usePointerIntent(host, { ease: 0.09 });

  // "Noticing you": a single short lift the first time the card is engaged.
  // It fires once, not on a loop â€” the brief is a mascot that reacts, not an
  // advertisement.
  useEffect(() => {
    if (!active || prefersReducedMotion()) return;
    setLifted(true);
    const t = window.setTimeout(() => setLifted(false), 420);
    return () => window.clearTimeout(t);
  }, [active]);

  const still = prefersReducedMotion();
  const rise = lifted && !still ? -7 : 0;

  return (
    <div
      ref={host}
      className={className}
      role="img"
      data-mascot="hands"
      aria-label="Two illustrated hands with index fingers raised. The 3D sign avatar performs recognized signs for you."
    >
      {/* viewBox 0 0 300 190: two ~110px hands plus a gap and room for their
          hard shadows, so neither hand is ever clipped.

          NOTE ON THE TRANSFORMS: `--mx` / `--my` are UNITLESS numbers (see
          usePointerIntent). That is what makes these expressions valid:
          `calc(-3deg + var(--mx) * 7deg)` adds angle to angle. An earlier
          version emitted `-0.42px`, which turned the same expression into
          "length + angle"; the browser dropped the whole transform without
          warning and both hands rendered stacked at the origin. */}
      <svg viewBox="0 0 300 190" className="h-full w-full" aria-hidden focusable="false">
        <g
          style={{
            transform: `translate(calc(var(--mx) * ${PX_SOFT}px), calc(var(--my) * ${PX_Y}px + ${rise}px))`,
            transition: still ? undefined : "transform 220ms cubic-bezier(0.2,0,0,1)",
          }}
        >
          {/* LEFT — lime, rotated -3deg, sitting higher. The two hands are NOT
              mirror images: different fill, angle, vertical offset and scale.
              Perfect symmetry is what makes mascot art look like clip art. */}
          <g
            style={{
              transform: `translate(calc(6px + var(--mx) * ${PX_SOFT}px), calc(-10px + var(--my) * ${PX_Y}px + ${rise}px)) rotate(calc(-3deg + var(--mx) * ${ROT_X}deg))`,
              transition: still ? undefined : "transform 140ms linear",
            }}
          >
            <Hand fill="var(--lime)" />
          </g>

          {/* RIGHT — sun, rotated +4deg, sitting lower and slightly larger. */}
          <g
            style={{
              transform: `translate(calc(150px + var(--mx) * ${PX_SOFT}px), calc(16px + var(--my) * ${PX_Y}px + ${rise}px)) scale(1.06) rotate(calc(4deg + var(--mx) * ${ROT_X}deg))`,
              transition: still ? undefined : "transform 140ms linear",
            }}
          >
            <Hand fill="var(--sun)" />
          </g>
        </g>
      </svg>
    </div>
  );
}
