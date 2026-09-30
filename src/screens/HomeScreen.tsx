import { useState, useRef } from "react";
import { BrutalButton } from "../components/ui/BrutalButton";
import { BrutalCard } from "../components/ui/BrutalCard";
import { BrutalBadge } from "../components/ui/BrutalBadge";
import { BrutalSticker } from "../components/ui/BrutalSticker";
import { BrutalLabel } from "../components/ui/BrutalLabel";
import { EyeMascot } from "../components/mascots/EyeMascot";
import { HandsMascot } from "../components/mascots/HandsMascot";
import { IconLogin, IconSpark } from "../components/icons";
import { useAuth } from "../auth/AuthProvider";
import type { AppView } from "../components/ui/BrutalNav";

/**
 * The HandSon wordmark. Drawn in markup (not an image) so it inherits the
 * current text colour, stays crisp at any size, and needs no extra asset.
 * The mark is a hand between two read-brackets.
 */
export function Wordmark({ className }: { className?: string }) {
  return (
    <span
      className={[
        "inline-flex items-center gap-2 font-[family-name:var(--font-display)] uppercase leading-none",
        className ?? "",
      ].join(" ")}
    >
      <svg
        viewBox="0 0 40 40"
        className="h-9 w-9 shrink-0"
        aria-hidden="true"
        fill="none"
        stroke="currentColor"
        strokeWidth={3.25}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M13 7 6 14v12l7 7" />
        <path d="M27 7l7 7v12l-7 7" />
        <path d="M20 26v-7a2 2 0 0 1 4 0v5" />
        <path d="M24 24v-3a2 2 0 0 1 4 0v7a8 8 0 0 1-8 8 9 9 0 0 1-7-4l-3-5a2 2 0 0 1 3-2l3 3" />
      </svg>
      <span>HandSon</span>
    </span>
  );
}

interface HomeScreenProps {
  onNavigate: (v: AppView) => void;
  /** Live model readiness, so the CTA never promises something unavailable. */
  modelReady: boolean;
  modelError: boolean;
}

/* ---------------------------------------------------------------------------
   CARD 01 â€” REAL-TIME TRANSLATOR
   Composition: a coral slab carries the number and the eye. The eye is given
   roughly a third of the card's width and BREAKS the slab's right edge, so the
   illustration is not a thing inside a box but a thing laid over one.
   ------------------------------------------------------------------------ */
function TranslatorCard({
  onNavigate,
  modelReady,
  modelError,
}: {
  onNavigate: (v: AppView) => void;
  modelReady: boolean;
  modelError: boolean;
}) {
  const [engaged, setEngaged] = useState(false);
  const ctaRef = useRef<HTMLDivElement | null>(null);

  return (
    <BrutalCard as="article" depth="xl" className="relative" aria-labelledby="feat-translate">
      {/* Colour slab. `overflow-visible` is the point: the eye hangs over it. */}
      <div className="relative overflow-visible border-b-[var(--border-w)] border-[var(--ink)] bg-[var(--coral)] px-4 pb-8 pt-5 sm:px-5">
        {/* Oversized numeral, OUTLINED, bleeding off the LEFT edge.
            It is placed in the empty band BELOW the heading, where it fills the
            gap the eye leaves on the left without ever sitting underneath the
            type. Behind it at z-0, and the slab keeps `overflow-visible` so the
            outline can bleed past the padding box.
            Placing it behind the heading was the earlier attempt; at 104px the
            counter of the "0" punched through between the glyphs and read as a
            stray curve rather than as a number. */}
        <span
          aria-hidden="true"
          className="hs-numeral pointer-events-none absolute -bottom-1 -left-1 z-0 text-[72px] opacity-55 sm:text-[88px]"
        >
          01
        </span>

        {/* Stickers overlap the slab's top border. */}
        <div className="relative z-10 flex flex-wrap items-start justify-between gap-2">
          <BrutalSticker tone="ink" rotate="-3" raised>
            <IconSpark className="h-3.5 w-3.5" aria-hidden />
            Live
          </BrutalSticker>
          <BrutalSticker tone="paper" rotate="3" raised>
            FSL
          </BrutalSticker>
        </div>

        {/* `pb-14` reserves the band the outlined "01" now occupies, so the
            heading never sits on top of it and the numeral never collides
            with the copy in the body below. */}
        <div className="relative z-10 mt-3 max-w-[62%] pb-14 sm:max-w-[58%] sm:pb-12">
          <BrutalLabel slashed tone="ink-on-bright">
            See it
          </BrutalLabel>
          <h2
            id="feat-translate"
            className="hs-mega m-0 mt-1 text-[clamp(2.1rem,9vw,3.1rem)]"
          >
            Real-Time
            <br />
            Translator
          </h2>
        </div>

        {/* The eye hangs off the right edge of the slab, overlapping the body
            below it. Negative right margin + z-index is the whole trick.
            `-bottom-4` (not -bottom-6) keeps the overlap shallow enough that it
            reads as layering rather than as an obstruction. */}
        <div
          className="pointer-events-auto absolute -bottom-3 right-[-16px] z-20 w-[38%] max-w-[168px] origin-bottom-right rotate-[3deg] sm:-bottom-4 sm:right-[-12px] sm:w-[38%] sm:max-w-[190px]"
          onPointerEnter={() => setEngaged(true)}
          onPointerLeave={() => setEngaged(false)}
        >
          <EyeMascot attentive={engaged} className="w-full" />
        </div>

        {/* Registration mark, bottom-left of the slab. */}
        <span aria-hidden="true" className="hs-mark-cross absolute bottom-2 left-3 opacity-50" />
      </div>

      {/* `pt-9` clears the eye's overlap into this block. Copy is capped at 62%
          so it never runs under the mascot, and the mascot is the only thing
          permitted to cross that boundary. */}
      <div className="relative px-4 pb-5 pt-9 sm:px-5 sm:pb-6 sm:pt-10">
        <p className="m-0 max-w-[62%] text-[15px] font-semibold leading-relaxed text-[var(--ink-soft)] sm:max-w-[66%]">
          Point your camera at a signer. HandSon reads Filipino Sign Language as
          they make it.
        </p>

        {/* Status is badge + words, never colour alone. */}
        <p
          className="m-0 mt-4 flex flex-wrap items-center gap-2 text-sm font-semibold text-[var(--ink-soft)]"
          role="status"
        >
          {modelError ? (
            <>
              <BrutalBadge tone="coral">Offline</BrutalBadge>
              <span>The FSL model could not load. Translation is unavailable.</span>
            </>
          ) : modelReady ? (
            <>
              <BrutalBadge tone="lime">Ready</BrutalBadge>
              <span>Model loaded. Camera starts on tap.</span>
            </>
          ) : (
            <>
              <BrutalBadge tone="paper">Loading</BrutalBadge>
              <span>Loading the Filipino Sign Language modelâ€¦</span>
            </>
          )}
        </p>

        <div ref={ctaRef} className="mt-5">
          <BrutalButton
            tone="ink"
            size="xl"
            block
            withArrow
            onClick={() => onNavigate("translate")}
            disabled={modelError}
          >
            {modelError ? "Model unavailable" : "Start translating"}
          </BrutalButton>
        </div>
      </div>
    </BrutalCard>
  );
}

/* ---------------------------------------------------------------------------
   CARD 02 â€” 3D SIGN AVATAR
   Deliberately NOT a mirror of card 01. It inverts on every axis:
     colour   lime instead of coral
     layout   heading right-aligned, mascot on the LEFT instead of the right
     shape    solid lime numeral block instead of an outlined numeral
   Related but distinct â€” two identical panels would read as a settings list.
   ------------------------------------------------------------------------ */
function AvatarCard({ onNavigate }: { onNavigate: (v: AppView) => void }) {
  const [engaged, setEngaged] = useState(false);

  return (
    <BrutalCard as="article" depth="xl" className="relative" aria-labelledby="feat-avatar">
      {/* `text-[#050505]` is set explicitly, exactly as on the coral slab in
          card 01. Pure --ink measures 4.34:1 on purple, which is legal for the
          oversized heading but not for the 10px kicker above it; near-black
          measures 4.69:1 and clears AA at every size on every bright fill. */}
      <div className="relative overflow-visible border-b-[var(--border-w)] border-[var(--ink)] bg-[var(--purple)] px-4 pb-8 pt-5 text-[#050505] sm:px-5">
        {/* Purple slab + lime numeral block. A solid shape, not an outline â€”
            the deliberate difference from card 01's stroked number. */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute -left-2 -top-3 z-10 rotate-[-4deg] border-[var(--border-w)] border-[var(--ink)] bg-[var(--lime)] px-3 py-1 font-[family-name:var(--font-mega)] text-[54px] leading-[0.8] tracking-tight text-[var(--ink)]"
        >
          02
        </span>

        <div className="relative z-10 flex flex-wrap items-start justify-end gap-2">
          <BrutalSticker tone="paper" rotate="-2" raised>
            Rotate
          </BrutalSticker>
          <BrutalSticker tone="ink" rotate="4" raised>
            3D
          </BrutalSticker>
        </div>

        <div className="relative z-10 mt-3 text-right">
          <BrutalLabel slashed tone="ink-on-bright">
            Sign it back
          </BrutalLabel>
          <h2 id="feat-avatar" className="hs-mega m-0 mt-1 text-[clamp(2.1rem,9vw,3.1rem)]">
            3D Sign
            <br />
            Avatar
          </h2>
        </div>

        {/* Hands break the LEFT edge â€” opposite side to the eye, which is what
            keeps the two cards from reading as one repeated component. */}
        <div
          className="pointer-events-auto absolute -bottom-3 left-[-14px] z-20 w-[38%] max-w-[168px] origin-bottom-left rotate-[-4deg] sm:-bottom-4 sm:left-[-10px] sm:w-[40%] sm:max-w-[190px]"
          onPointerEnter={() => setEngaged(true)}
          onPointerLeave={() => setEngaged(false)}
        >
          <HandsMascot active={engaged} className="w-full" />
        </div>

        <span aria-hidden="true" className="hs-mark-cross absolute bottom-2 right-3 opacity-60" />
      </div>

      <div className="relative px-4 pb-5 pt-9 sm:px-5 sm:pb-6 sm:pt-10">
        <p className="m-0 ml-auto max-w-[64%] text-right text-[15px] font-semibold leading-relaxed text-[var(--ink-soft)] sm:max-w-[66%]">
          Type a phrase and the HandSon avatar performs it, so you can study the
          shape of a sign.
        </p>

        <div className="mt-5">
          <BrutalButton
            tone="purple"
            size="xl"
            block
            withArrow
            onClick={() => onNavigate("avatar")}
          >
            Open 3D avatar
          </BrutalButton>
        </div>
      </div>
    </BrutalCard>
  );
}

/* ---------------------------------------------------------------------------
   HERO
   The brief for this section, in order of visual weight:
     upper-left   small `// FILIPINO SIGN LANGUAGE` kicker
     centre-left  GIANT "Make signs visible." in Anton, three lines, tight
     lower-left   supporting copy
     upper-right  a sticker that overhangs the page edge
     lower-right  a colour block, breaking the type's right margin

   Asymmetry is structural, not decorative: the type column is capped at 68%
   and the graphic column occupies the rest, so the two never form a symmetric
   pair at any breakpoint.
   ------------------------------------------------------------------------ */
function Hero({
  firstName,
  signedIn,
  onAccount,
}: {
  firstName: string | null;
  signedIn: boolean;
  onAccount: () => void;
}) {
  return (
    <section className="relative mb-9 grid grid-cols-1 gap-x-10 sm:mb-12 lg:grid-cols-[minmax(0,1fr)_320px]" aria-labelledby="hero-title">
      {/* HEADER spans both columns, then everything below splits into a type
          column and a graphic column. `grid` (not flex) so the graphic column
          keeps a real width the type can be measured against; the type column
          takes the remainder. */}
      <header className="col-span-full mb-6 flex items-start justify-between gap-3 sm:mb-8">
        <Wordmark className="hs-rot-neg1 text-2xl sm:text-3xl" />
        <BrutalButton tone="paper" size="sm" onClick={onAccount} className="shrink-0">
          <IconLogin className="h-4 w-4" aria-hidden />
          {signedIn ? "Account" : "Log in"}
        </BrutalButton>
      </header>

      {/* LEFT COLUMN — type, then supporting copy. */}
      <div className="min-w-0">
      {/* Kicker + oversized headline. The three words are on separate lines with
          an explicit break so the block has a poster silhouette.

          `sm:max-w-none` lets the type use the left column of the grid below;
          the cap is a MOBILE-only measure, because at 19ch the headline keeps
          a poster silhouette instead of running the full width of the screen. */}
      <div className="relative max-w-[19ch] sm:max-w-none">
        <div className="flex flex-wrap items-center gap-2">
          <BrutalLabel slashed tone="ink">
            Filipino Sign Language
          </BrutalLabel>
          {firstName ? (
            <BrutalSticker tone="lime" rotate="-2">
              Hello, {firstName}
            </BrutalSticker>
          ) : (
            /* The greeting is shown either way. An anonymous visitor gets the
               same welcome, so the page never opens on a bare headline. */
            <BrutalSticker tone="lime" rotate="-2">
              Hello there
            </BrutalSticker>
          )}
        </div>

        <h1
          id="hero-title"
          className="hs-mega m-0 mt-3 text-[clamp(3.1rem,15.5vw,7.5rem)] sm:mt-4"
        >
          Make
          <br />
          Signs
          <br />
          <span className="relative inline-block">
            Visible.
            {/* Underline slab: a solid ink bar offset behind the word, which is
                what makes the type feel printed rather than typed. */}
            <span
              aria-hidden="true"
              className="absolute -bottom-1 left-0 -z-10 h-[0.34em] w-[104%] -rotate-[1.5deg] bg-[var(--sun)]"
            />
          </span>
        </h1>
      </div>

      {/* Lower band: copy on the left, colour block on the right.
          `sm:items-end sm:justify-between` creates the asymmetry on wider
          screens. On mobile the block sits UNDER the copy as a separate flex
          item, so the paragraph keeps its full measure instead of being
          squeezed to ~60% of an already narrow column. */}
      <div className="mt-6 flex flex-col items-start gap-4 sm:mt-7 sm:flex-row sm:items-end sm:justify-between sm:gap-6">
        <p className="m-0 max-w-[34ch] text-[15px] font-semibold leading-relaxed text-[var(--ink-soft)] sm:text-base">
          HandSon turns Filipino Sign Language into something you can see,
          understand, and interact with — in real time, and in 3D.
        </p>

        {/* Colour block. Shown at ALL widths, but scaled down on mobile: the
            brief wants asymmetry on every breakpoint, and hiding it entirely
            below `sm` left the small screen as a plain centred column. On
            mobile it sits inline after the copy rather than in a flex row, so
            it never squeezes the paragraph. */}
        <div
          aria-hidden="true"
          className="hs-rot-2 relative h-[58px] w-[132px] shrink-0 border-[var(--border-w)] border-[var(--ink)] bg-[var(--teal)] shadow-[var(--shadow-sm)] sm:h-[74px] sm:w-[190px]"
        >
          <span className="absolute left-2 top-1.5 font-[family-name:var(--font-display)] text-[9px] uppercase tracking-[0.2em] text-[var(--ink)] sm:top-2 sm:text-[10px]">
            Est. 2026
          </span>
          <span className="absolute bottom-1 right-2 font-[family-name:var(--font-mega)] text-[30px] leading-none text-[var(--ink)] sm:text-[40px]">
            3D
          </span>
          <span className="hs-dots absolute right-1.5 top-1.5 h-2.5 w-2.5 opacity-40 sm:h-3 sm:w-3" />
        </div>
      </div>
      {/* end LEFT COLUMN — type and supporting copy */}
      </div>

      {/* RIGHT COLUMN — the hero's graphic weight, desktop only.
          At lg the left column is type and this column is a stack of printed
          objects, so the page reads as a SPREAD rather than a centred column
          floating in empty space. This is where the brief's asymmetry lives:
          nothing here is mirrored against the type, and the blocks sit at three
          different heights and two different rotations.

          Hidden below lg: at 414px there is no room beside a 7.5rem headline
          without shrinking the type, and the two stacked cards below already
          carry the colour weight at that size. */}
      <div aria-hidden="true" className="relative hidden lg:block">
        {/* Lime slab with a numeral, rotated, high-right. */}
        <div className="hs-rot-2 absolute right-0 top-0 flex h-[122px] w-[152px] items-center justify-center border-[var(--border-w)] border-[var(--ink)] bg-[var(--lime)] shadow-[var(--shadow)]">
          <span className="font-[family-name:var(--font-mega)] text-[92px] leading-none text-[var(--ink)]">
            08
          </span>
        </div>

        {/* Coral bar crossing beneath it, on a different rotation. */}
        <div className="hs-rot-neg1 absolute right-[128px] top-[88px] h-[34px] w-[190px] border-[var(--border-w)] border-[var(--ink)] bg-[var(--coral)] shadow-[var(--shadow-sm)]" />

        {/* Teal rule lower down, with a halftone block beneath. */}
        <div className="absolute right-0 top-[172px] h-[46px] w-[214px] border-[var(--border-w)] border-[var(--ink)] bg-[var(--teal)] shadow-[var(--shadow-sm)]">
          <span className="absolute left-2 top-1.5 font-[family-name:var(--font-display)] text-[10px] uppercase tracking-[0.2em] text-[var(--ink)]">
            Two ways in
          </span>
        </div>
        <div className="hs-dots absolute right-[4px] top-[230px] h-6 w-6 opacity-35" />

        {/* Registration mark, pulled off the column's left edge. */}
        <span className="hs-mark-cross absolute -left-7 top-3 opacity-40" />
      </div>

      {/* Registration marks, placed asymmetrically on purpose. */}
      <span aria-hidden="true" className="hs-mark-cross absolute -right-1 top-16 hidden opacity-40 lg:block" />
      <span aria-hidden="true" className="hs-mark-cross absolute left-1/2 top-4 hidden opacity-30 lg:block" />
    </section>
  );
}


/* ---------------------------------------------------------------------------
   HOME
   The container is deliberately WIDE (max-w-3xl, not 2xl) and the page is NOT
   centred: it is pulled slightly left of centre so the hero's right margin can
   carry the colour block. Stacking is preserved — cards 01 and 02 are
   `article` elements in document order, one above the other.
   ------------------------------------------------------------------------ */
export function HomeScreen({ onNavigate, modelReady, modelError }: HomeScreenProps) {
  const { user } = useAuth();
  const firstName = user?.name?.trim().split(/\s+/)[0] ?? null;

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6">
      <Hero
        firstName={firstName}
        signedIn={Boolean(user)}
        onAccount={() => onNavigate("account")}
      />

      {/* Section rule with an oversized index, the editorial device that ties
          the hero to the two cards. */}
      <div className="mb-5 flex items-center gap-3 sm:mb-6">
        <BrutalLabel tone="mute">Explore</BrutalLabel>
        <span aria-hidden="true" className="h-[3px] flex-1 bg-[var(--ink)]" />
      </div>

      {/* Stacked. `space-y` is used instead of margins so the two articles are
          guaranteed non-overlapping siblings. */}
      <div className="space-y-7 sm:space-y-9">
        <TranslatorCard
          onNavigate={onNavigate}
          modelReady={modelReady}
          modelError={modelError}
        />
        <AvatarCard onNavigate={onNavigate} />
      </div>
    </div>
  );
}

