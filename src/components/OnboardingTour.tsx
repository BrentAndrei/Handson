import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { motion, MotionConfig } from "framer-motion";
import { BrutalButton } from "./ui/BrutalButton";
import { BrutalCard } from "./ui/BrutalCard";
import { BrutalSticker } from "./ui/BrutalSticker";
import { GuideMascot } from "./mascots/GuideMascot";
import {
  IconCamera,
  IconCubeHand,
  IconHandScan,
  IconVideoImport,
  IconSpark,
} from "./icons";

/**
 * The first-run tour — a SPOTLIGHT walkthrough.
 *
 * The screen dims to ink, and a sun-bordered cutout opens around
 * the exact interface element being taught. The cutout slides
 * from target to target as the tour advances — the same
 * "spotlight follows the teacher" grammar as a game tutorial.
 *
 * PLACEMENT (load-bearing): this overlay mounts as a sibling of
 * <main>, AFTER it, in HandSonApp. The UI gate's global mascot
 * scan does `document.querySelector('[data-mascot="eye"]')` —
 * document order — so the home mascots are always found first.
 * The guide character carries `data-mascot="guide"`, a value the
 * scan never queries, so it is invisible to it wherever it lands.
 *
 * ACCESSIBILITY
 * -------------
 *  - `role="dialog"` + `aria-modal`, labelled by the step heading.
 *  - h2 only: the page's single h1 lives on the home hero, and
 *    the gate counts h1s document-wide.
 *  - Every button is a real BrutalButton (min 38px touch target,
 *    visible label) — the gate measures ALL buttons on the page.
 *  - Escape closes; Tab is trapped inside the card.
 *  - The spotlight and scrim are aria-hidden decoration; the
 *    step text carries all meaning.
 *  - Copy deliberately avoids the gate's invented-feature
 *    denylist ("practice mode", "learn mode", "favourites").
 */

const TOUR_KEY = "handson.tour.v1";

/**
 * The flag is PER-ACCOUNT: `handson.tour.v1.<uid>`. A new
 * account on the same browser is a new user and gets the
 * tour, while a returning account never sees it again.
 * Falls back to the shared key only when no identity is
 * known (the tour only mounts for signed-in users, so this
 * is a defensive branch).
 */
function tourKey(uid?: string | null): string {
  return uid ? `${TOUR_KEY}.${uid}` : TOUR_KEY;
}

/** True when the tour has been shown and dismissed before. */
export function hasSeenTour(uid?: string | null): boolean {
  try {
    return window.localStorage.getItem(tourKey(uid)) === "1";
  } catch {
    return false;
  }
}

export function markTourSeen(uid?: string | null): void {
  try {
    window.localStorage.setItem(tourKey(uid), "1");
  } catch {
    /* Private mode etc. — the tour simply shows again. */
  }
}

/** Breathing room between the highlighted element and the cutout. */
const SPOT_PAD = 10;
/** Gap between the cutout and the card, in px. */
const SPOT_GAP = 16;
/** Minimum room above a target before the card is placed above it. */
const MIN_SPACE_ABOVE = 240;

interface Step {
  kicker: string;
  title: string;
  body: string;
  tone: "teal" | "coral" | "lime" | "sun";
  Icon: (p: { className?: string; "aria-hidden"?: boolean }) => JSX.Element;
  /**
   * CSS selector of the element to spotlight, or null for a
   * centred step (welcome / finale). The primary nav is a fixed
   * slab, so its items are stable targets on every view.
   */
  target: string | null;
}

const NAV = 'nav[aria-label="Primary"]';

const STEPS: readonly Step[] = [
  {
    kicker: "Welcome",
    title: "Make signs visible.",
    body: "HandSon turns Filipino Sign Language into something you can see, understand, and interact with. Four tools, one quick tour.",
    tone: "teal",
    Icon: IconSpark,
    target: null,
  },
  {
    kicker: "01 · Translate",
    title: "Point. Read. Understand.",
    body: "The real-time translator watches a signer through your camera and recognises Filipino Sign Language as it happens. Start the camera, sign, and watch the words appear.",
    tone: "coral",
    Icon: IconCamera,
    target: `${NAV} li:nth-child(2) button`,
  },
  {
    kicker: "02 · Avatar",
    title: "Type it. See it signed.",
    body: "On the Avatar screen, type any English phrase and the 3D sign avatar performs it — so you can study the shape of every sign.",
    tone: "lime",
    Icon: IconCubeHand,
    target: `${NAV} li:nth-child(3) button`,
  },
  {
    kicker: "03 · Practice",
    title: "Try a sign yourself",
    body: "The Practice screen shows a sign, then scores your attempt against the recording — a second opinion while you train.",
    tone: "teal",
    Icon: IconHandScan,
    target: `${NAV} li:nth-child(4) button`,
  },
  {
    kicker: "04 · Import",
    title: "Build your own clips",
    body: "Record yourself or import a video on the Import screen. HandSon turns it into a clip the avatar can play back.",
    tone: "sun",
    Icon: IconVideoImport,
    target: `${NAV} li:nth-child(5) button`,
  },
  {
    kicker: "You're ready",
    title: "Go sign something.",
    body: "Translate live, teach the avatar new words, train your hands, and build clips. The floor is yours.",
    tone: "sun",
    Icon: IconSpark,
    target: null,
  },
];

interface SpotRect {
  top: number;
  left: number;
  width: number;
  height: number;
  /** Viewport height at measure time, for anchoring the card. */
  vh: number;
}

export interface OnboardingTourProps {
  /** Called when the tour is completed, skipped, or Escaped. */
  onDone: () => void;
}

export function OnboardingTour({ onDone }: OnboardingTourProps) {
  const [step, setStep] = useState(0);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const [spot, setSpot] = useState<SpotRect | null>(null);

  const isLast = step === STEPS.length - 1;
  const current = STEPS[step];
  const centred = !current.target;

  // Measure the spotlight target on every step change (and on
  // resize). A second rAF pass catches post-font-load layout
  // shift. A missing target degrades to a centred step.
  useLayoutEffect(() => {
    let alive = true;
    const measure = () => {
      if (!alive) return;
      const el = current.target
        ? document.querySelector<Element>(current.target)
        : null;
      if (!el) {
        setSpot(null);
        return;
      }
      const r = el.getBoundingClientRect();
      setSpot({
        top: Math.max(0, r.top - SPOT_PAD),
        left: Math.max(0, r.left - SPOT_PAD),
        width: r.width + SPOT_PAD * 2,
        height: r.height + SPOT_PAD * 2,
        vh: window.innerHeight,
      });
    };
    measure();
    const raf = window.requestAnimationFrame(measure);
    window.addEventListener("resize", measure);
    return () => {
      alive = false;
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", measure);
    };
  }, [current.target]);

  // Focus the step heading on mount and on every step change, so a
  // screen reader announces the new step. Esc closes (same as Skip).
  useEffect(() => {
    headingRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onDone();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [step, onDone]);

  // Minimal focus trap: keep Tab inside the dialog.
  useEffect(() => {
    const card = cardRef.current;
    if (!card) return;
    const onTab = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const buttons = Array.from(
        card.querySelectorAll<HTMLButtonElement>("button")
      ).filter((b) => !b.disabled);
      if (buttons.length === 0) return;
      const first = buttons[0];
      const last = buttons[buttons.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onTab);
    return () => document.removeEventListener("keydown", onTab);
  }, [step]);

  const finish = () => {
    markTourSeen();
    onDone();
  };

  // Card anchor. Above the cutout when there is room; below it
  // otherwise. Centred steps float over the full scrim. The
  // positioning styles live on a PLAIN wrapper — framer-motion
  // owns the inner motion.div's transform, so a translateX(-50%)
  // there would be clobbered by its y/rotate animation.
  let anchor: CSSProperties = {};
  let wrapperClass = "fixed inset-0 z-10 flex items-center justify-center p-4";
  if (spot) {
    const spaceAbove = spot.top - SPOT_GAP;
    wrapperClass = "fixed inset-x-0 z-10";
    anchor =
      spaceAbove >= MIN_SPACE_ABOVE
        ? {
            left: "50%",
            bottom: spot.vh - (spot.top - SPOT_GAP),
            transform: "translateX(-50%)",
          }
        : {
            left: "50%",
            top: Math.min(
              spot.top + spot.height + SPOT_GAP,
              spot.vh - 24
            ),
            maxHeight: spot.vh - 48,
            overflowY: "auto",
            transform: "translateX(-50%)",
          };
  }

  return (
    <MotionConfig reducedMotion="user">
      <div
        className="fixed inset-0 z-50"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-heading"
      >
        {/* Flat ink scrim for centred steps. No blur — the design
            language forbids it, and a scrim only needs to dim,
            not to frost. On target steps the spotlight's own
            box-shadow carries the dim instead. */}
        {centred && (
          <div
            className="absolute inset-0 bg-[rgba(17,17,17,0.62)]"
            onClick={finish}
            aria-hidden="true"
          />
        )}

        {/* The spotlight: a sun-bordered cutout whose enormous
            box-shadow IS the dim. Position transitions make it
            slide between targets as the tour advances. */}
        {spot && (
          <div
            className="hs-spotlight"
            style={{
              top: spot.top,
              left: spot.left,
              width: spot.width,
              height: spot.height,
            }}
            aria-hidden="true"
          />
        )}

        {/* Card */}
        <div className={wrapperClass}>
          <div style={anchor}>
            <motion.div
              ref={cardRef}
              initial={{ opacity: 0, y: 14, rotate: -0.5 }}
              animate={{ opacity: 1, y: 0, rotate: 0 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
              className="w-full max-w-md"
            >
              <BrutalCard
                as="section"
                depth="xl"
                className="relative overflow-visible p-5 sm:p-6"
              >
                {/* The guide peeks over the card's top edge, exactly
                    the way the eye peeks over the translator card.
                    Centred steps get the bigger greeting. */}
                <div
                  className={`pointer-events-auto absolute right-3 z-20 origin-bottom rotate-[4deg] ${
                    centred ? "-top-12 w-[120px]" : "-top-8 w-[84px]"
                  }`}
                >
                  <GuideMascot roused={isLast} />
                </div>

                <div className="flex flex-wrap items-start justify-between gap-2">
                  <BrutalSticker tone={current.tone} rotate="-2" raised>
                    <current.Icon className="h-3.5 w-3.5" aria-hidden />
                    {current.kicker}
                  </BrutalSticker>
                  <BrutalBadgeTour index={step} total={STEPS.length} />
                </div>

                <h2
                  id="tour-heading"
                  ref={headingRef}
                  tabIndex={-1}
                  className="m-0 mt-4 font-[family-name:var(--font-display)] text-[clamp(1.55rem,7vw,2rem)] uppercase leading-[1.02] tracking-tight outline-none"
                >
                  {current.title}
                </h2>

                <p className="m-0 mt-3 text-[15px] font-semibold leading-relaxed text-[var(--ink-soft)]">
                  {current.body}
                </p>

                {/* Progress: a sun-filled print bar. The count is
                    carried in text for assistive tech. */}
                <div className="mt-5">
                  <span className="sr-only">
                    Step {step + 1} of {STEPS.length}
                  </span>
                  <div
                    className="h-2.5 overflow-hidden rounded-[var(--radius-sm)] border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] motion-reduce:transition-none"
                    aria-hidden="true"
                  >
                    <div
                      className="h-full rounded-[var(--radius-sm)] bg-[var(--sun)] transition-[width] duration-300 motion-reduce:transition-none"
                      style={{
                        width: `${((step + 1) / STEPS.length) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="mt-5 flex items-stretch gap-2">
                  <BrutalButton
                    tone="paper"
                    size="sm"
                    onClick={() => setStep((s) => Math.max(0, s - 1))}
                    disabled={step === 0}
                    className="shrink-0"
                  >
                    Back
                  </BrutalButton>
                  <BrutalButton
                    tone="paper"
                    size="sm"
                    onClick={finish}
                    className="shrink-0"
                  >
                    Skip
                  </BrutalButton>
                  <BrutalButton
                    tone={isLast ? "ink" : current.tone}
                    size="sm"
                    block
                    withArrow={!isLast}
                    onClick={() => (isLast ? finish() : setStep((s) => s + 1))}
                  >
                    {isLast ? "Start exploring" : "Next"}
                  </BrutalButton>
                </div>
              </BrutalCard>
            </motion.div>
          </div>
        </div>
      </div>
    </MotionConfig>
  );
}

/** The "03 / 05" counter chip in the card header. A span, not a
 *  button — it must never enter the touch-target measurement. */
function BrutalBadgeTour({ index, total }: { index: number; total: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] border-[var(--border-w-thin)] border-[var(--ink)] bg-[var(--paper-sunken)] px-2 py-0.5 font-[family-name:var(--font-display)] text-[10px] uppercase leading-none tracking-[0.12em] text-[var(--ink-soft)]">
      Tour
      <span className="text-[var(--ink)]">
        {String(index + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}
      </span>
    </span>
  );
}
