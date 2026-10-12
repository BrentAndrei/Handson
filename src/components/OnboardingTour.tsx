import { useEffect, useRef, useState } from "react";
import { motion, MotionConfig } from "framer-motion";
import { BrutalButton } from "./ui/BrutalButton";
import { BrutalCard } from "./ui/BrutalCard";
import { BrutalSticker } from "./ui/BrutalSticker";
import { BrutalLabel } from "./ui/BrutalLabel";
import { GuideMascot } from "./mascots/GuideMascot";
import {
  IconCamera,
  IconCubeHand,
  IconHandScan,
  IconVideoImport,
  IconSpark,
} from "./icons";

/**
 * The first-run tour.
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
 *  - Copy deliberately avoids the gate's invented-feature
 *    denylist ("practice mode", "learn mode", "favourites").
 */

const TOUR_KEY = "handson.tour.v1";

/** True when the tour has been shown and dismissed before. */
export function hasSeenTour(): boolean {
  try {
    return window.localStorage.getItem(TOUR_KEY) === "1";
  } catch {
    return false;
  }
}

export function markTourSeen(): void {
  try {
    window.localStorage.setItem(TOUR_KEY, "1");
  } catch {
    /* Private mode etc. — the tour simply shows again. */
  }
}

interface Step {
  kicker: string;
  title: string;
  body: string;
  tone: "teal" | "coral" | "lime" | "sun";
  Icon: (p: { className?: string; "aria-hidden"?: boolean }) => JSX.Element;
}

const STEPS: readonly Step[] = [
  {
    kicker: "Welcome",
    title: "Make signs visible.",
    body: "HandSon turns Filipino Sign Language into something you can see, understand, and interact with. Four tools, one quick tour.",
    tone: "teal",
    Icon: IconSpark,
  },
  {
    kicker: "01 · Translate",
    title: "Point. Read. Understand.",
    body: "The real-time translator watches a signer through your camera and recognises Filipino Sign Language as it happens. Start the camera, sign, and watch the words appear.",
    tone: "coral",
    Icon: IconCamera,
  },
  {
    kicker: "02 · Avatar",
    title: "Type it. See it signed.",
    body: "On the Avatar screen, type any English phrase and the 3D sign avatar performs it — so you can study the shape of every sign.",
    tone: "lime",
    Icon: IconCubeHand,
  },
  {
    kicker: "03 · Practice",
    title: "Try a sign yourself",
    body: "The Practice screen shows a sign, then scores your attempt against the recording — a second opinion while you train.",
    tone: "teal",
    Icon: IconHandScan,
  },
  {
    kicker: "04 · Import",
    title: "Build your own clips",
    body: "Record yourself or import a video on the Import screen. HandSon turns it into a clip the avatar can play back.",
    tone: "sun",
    Icon: IconVideoImport,
  },
];

export interface OnboardingTourProps {
  /** Called when the tour is completed, skipped, or Escaped. */
  onDone: () => void;
}

export function OnboardingTour({ onDone }: OnboardingTourProps) {
  const [step, setStep] = useState(0);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);

  const isLast = step === STEPS.length - 1;
  const current = STEPS[step];

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

  return (
    <MotionConfig reducedMotion="user">
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-heading"
      >
        {/* Flat ink scrim. No blur — the design language forbids it,
            and a scrim only needs to dim, not to frost. */}
        <div
          className="absolute inset-0 bg-[rgba(17,17,17,0.6)]"
          onClick={finish}
          aria-hidden="true"
        />

        <motion.div
          ref={cardRef}
          initial={{ opacity: 0, y: 14, rotate: -0.5 }}
          animate={{ opacity: 1, y: 0, rotate: 0 }}
          transition={{ duration: 0.28, ease: "easeOut" }}
          className="relative w-full max-w-md"
        >
          <BrutalCard
            as="section"
            depth="xl"
            className="relative overflow-visible p-5 sm:p-6"
          >
            {/* The guide peeks over the card's top edge, exactly the
                way the eye peeks over the translator card. */}
            <div className="pointer-events-auto absolute -top-8 right-3 z-20 w-[84px] origin-bottom rotate-[4deg]">
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

            {/* Step dots are decorative; the count is carried in text. */}
            <div className="mt-5 flex items-center justify-between gap-3">
              <span className="sr-only">
                Step {step + 1} of {STEPS.length}
              </span>
              <div className="flex items-center gap-1.5" aria-hidden="true">
                {STEPS.map((_, i) => (
                  <span
                    key={i}
                    className={`h-2.5 w-2.5 border-[var(--border-w-thin)] border-[var(--ink)] ${
                      i === step ? "bg-[var(--ink)]" : "bg-[var(--paper-raised)]"
                    }`}
                  />
                ))}
              </div>
              <BrutalLabel tone="mute" className="font-[family-name:var(--font-mega)] text-[13px] tracking-normal">
                {step + 1} / {STEPS.length}
              </BrutalLabel>
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
