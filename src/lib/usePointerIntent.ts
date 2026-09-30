import { useEffect, useRef } from "react";

/**
 * Shared pointer + scroll intent for the home mascots.
 *
 * WHY A HOOK AND NOT PER-COMPONENT STATE
 * --------------------------------------
 * Both mascots need "where is the user pointing" and "which way is the page
 * moving". React state would re-render 60x/second and fight the existing
 * Three.js loop. Instead this writes CSS custom properties and lets the
 * compositor do the work, so nothing re-renders after mount.
 *
 * PERFORMANCE RULES ENFORCED HERE
 *  - ONE rAF loop, which self-parks when the value has settled.
 *  - Listeners are `{ passive: true }`; scrolling is never blocked.
 *  - `pointermove` is read once per frame, not once per event.
 *  - `IntersectionObserver` gates everything: an off-screen mascot observes
 *    nothing and holds no listeners.
 *  - `prefers-reduced-motion` never starts a loop at all.
 *
 * Emitted values are NORMALISED and CLAMPED to [-1, 1]:
 *    x: -1 pointer hard left, +1 hard right.  y: -1 above, +1 below.
 * A consumer multiplies by whatever travel its art can afford, so this clamp
 * is the only thing between a cursor and a runaway transform.
 *
 * EMITS UNITLESS NUMBERS, NOT LENGTHS. `--mx` is `-0.42`, never `-0.42px`.
 * The mascots need to multiply this into BOTH a px translate and a degree
 * rotation. If the property carried a unit, `var(--mx) / 14 * 7deg` would be a
 * length divided by a number — still a length — and adding a length to an
 * angle is invalid CSS. The browser then drops the whole `transform`
 * declaration with no error, and both hands silently render stacked at the
 * origin. Emitting a plain number keeps every consumer valid:
 *     translate(calc(146px + var(--mx) * 26px), ...) rotate(calc(4deg + var(--mx) * 7deg))
 */
export function usePointerIntent(
  ref: React.RefObject<HTMLElement | null>,
  options: {
    /**
     * How fast the value chases the pointer. 1 = instant, 0.1 = dreamy.
     * Travel distances are NOT configured here — the emitted value is
     * unitless, so each mascot picks its own px and deg multipliers.
     */
    ease?: number;
    /** Below this magnitude the loop parks itself. */
    epsilon?: number;
  } = {}
) {
  const { ease = 0.12, epsilon = 0.0025 } = options;

  // Tunables live in a ref so changing them never restarts the loop.
  const cfg = useRef({ ease, epsilon });
  cfg.current = { ease, epsilon };

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    /* Seed the custom properties IMMEDIATELY, before the first frame and
       before the effect body can bail out.
       This is not cosmetic. A custom property that is not set makes every
       `calc(var(--mx) * 14px)` in a descendant INVALID, because the fallback in
       `var(--mx, 0)` is only used when the property is *unset* — and inside
       `calc()` an unregistered custom property with no fallback makes the whole
       declaration invalid at computed-value time. The browser then drops the
       `transform` entirely and both mascots render unpositioned, with no error
       anywhere. Seeding them here means the resting state is always valid CSS. */
    for (const prop of ["--mx", "--my", "--mx-soft", "--my-soft"]) {
      el.style.setProperty(prop, "0");
    }

    // Reduced motion => completely static. The component's resting transform
    // is the correct reduced-motion state, so we simply never start a loop.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let pointerX = window.innerWidth / 2;
    let pointerY = window.innerHeight / 2;
    let curX = 0;
    let curY = 0;
    /** Decaying scroll impulse, in px. */
    let scrollImpulse = 0;
    let lastScrollY = window.scrollY;
    let visible = false;
    let attached = false;
    let raf = 0;

    const clamp = (v: number) => (v < -1 ? -1 : v > 1 ? 1 : v);

    /** Re-derive the goal from the latest pointer position. */
    function measure() {
      const box = el!.getBoundingClientRect();
      targetX = clamp((pointerX - (box.left + box.width / 2)) / (window.innerWidth / 2));
      targetY = clamp((pointerY - (box.top + box.height / 2)) / (window.innerHeight / 2));
    }
    let targetX = 0;
    let targetY = 0;

    function frame() {
      const { ease: e, epsilon: eps } = cfg.current;
      // Scroll adds a transient vertical bias that decays on its own, so the
      // mascot glances toward incoming content, then returns to neutral.
      const glide = scrollImpulse * 0.0016;
      const goalX = targetX;
      const goalY = clamp(targetY + glide);

      curX += (goalX - curX) * e;
      curY += (goalY - curY) * e;
      scrollImpulse *= 0.9;

      // Unitless, 4dp. See the note at the top of this file: these must stay
      // unit-free so consumers can use them in both px and deg contexts.
      el!.style.setProperty("--mx", curX.toFixed(4));
      el!.style.setProperty("--my", curY.toFixed(4));
      el!.style.setProperty("--mx-soft", curX.toFixed(4));
      el!.style.setProperty("--my-soft", curY.toFixed(4));

      const settled =
        Math.abs(goalX - curX) < eps &&
        Math.abs(goalY - curY) < eps &&
        Math.abs(scrollImpulse) < 1;

      if (settled) {
        // Park. The next pointer or scroll event restarts us.
        raf = 0;
        return;
      }
      raf = requestAnimationFrame(frame);
    }

    function kick() {
      if (visible && raf === 0) raf = requestAnimationFrame(frame);
    }

    function onPointer(e: PointerEvent) {
      pointerX = e.clientX;
      pointerY = e.clientY;
      measure();
      kick();
    }

    function onScroll() {
      const y = window.scrollY;
      scrollImpulse += y - lastScrollY;
      lastScrollY = y;
      kick();
    }

    function onTouch(e: TouchEvent) {
      const t0 = e.touches[0];
      if (!t0) return;
      pointerX = t0.clientX;
      pointerY = t0.clientY;
      measure();
      kick();
    }

    function attach() {
      if (attached) return;
      attached = true;
      window.addEventListener("pointermove", onPointer, { passive: true });
      window.addEventListener("scroll", onScroll, { passive: true });
      window.addEventListener("touchmove", onTouch, { passive: true });
    }

    function detach() {
      if (!attached) return;
      attached = false;
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("touchmove", onTouch);
    }

    const io = new IntersectionObserver(
      (entries) => {
        visible = entries.some((entry) => entry.isIntersecting);
        if (visible) {
          attach();
          measure();
          kick();
        } else {
          if (raf) cancelAnimationFrame(raf);
          raf = 0;
          detach();
        }
      },
      { threshold: 0.15 }
    );
    io.observe(el);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      io.disconnect();
      detach();
    };
  }, [ref]);
}

/**
 * Current `prefers-reduced-motion` state. Read imperatively by mascots that
 * need a boolean outside the render pass.
 */
export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
