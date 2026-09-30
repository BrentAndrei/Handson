/**
 * HANDSON ICON SET
 * ---------------------------------------------------------------------------
 * Hand-authored rather than pulled from a package, for two reasons:
 *  1. Zero runtime dependency — the project has no icon library installed.
 *  2. One guaranteed visual language. Every glyph below uses the SAME geometry:
 *     a 24-unit box, 2.25 stroke, round caps and joins, no fills. Mixing
 *     libraries is how UIs end up with mismatched stroke weights.
 *
 * All icons are decorative by default (`aria-hidden`); pass a `title` to expose
 * one to assistive tech.
 */
import type { SVGProps } from "react";

export type IconProps = SVGProps<SVGSVGElement> & {
  /** Accessible name. When omitted the icon is hidden from screen readers. */
  title?: string;
};

function Icon({ title, children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.25}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={title ? undefined : true}
      role={title ? "img" : undefined}
      focusable="false"
      {...props}
    >
      {title ? <title>{title}</title> : null}
      {children}
    </svg>
  );
}

/* --- feature + navigation ------------------------------------------------ */

/** Home: a house reduced to its frame. */
export const IconHome = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 10.5 12 3l9 7.5" />
    <path d="M5 9.5V20h14V9.5" />
    <path d="M10 20v-6h4v6" />
  </Icon>
);

/** Translate: a camera body with a viewfinder — the capture act. */
export const IconCamera = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 8.5h4l2-2.5h6l2 2.5h4v11H3z" />
    <circle cx="12" cy="13.5" r="3.5" />
  </Icon>
);

/** Avatar: a head-and-shoulders mark, drawn as an outline, not a robot. */
export const IconAvatar = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="8" r="4" />
    <path d="M4.5 20.5c0-4 3.4-6.5 7.5-6.5s7.5 2.5 7.5 6.5" />
  </Icon>
);

/** Account: a person, deliberately abstract. */
export const IconUser = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="8" r="3.75" />
    <path d="M5 20.5c0-3.9 3.1-6.25 7-6.25s7 2.35 7 6.25" />
  </Icon>
);

/* --- auth ----------------------------------------------------------------- */

export const IconLogin = (p: IconProps) => (
  <Icon {...p}>
    <path d="M10 4H5v16h5" />
    <path d="M14 8l4 4-4 4" />
    <path d="M18 12H9" />
  </Icon>
);

export const IconLogout = (p: IconProps) => (
  <Icon {...p}>
    <path d="M14 4h5v16h-5" />
    <path d="M10 8l-4 4 4 4" />
    <path d="M6 12h9" />
  </Icon>
);

/* --- utility -------------------------------------------------------------- */

export const IconSettings = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="3" />
    <path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.2 5.2l2.1 2.1M16.7 16.7l2.1 2.1M18.8 5.2l-2.1 2.1M7.3 16.7l-2.1 2.1" />
  </Icon>
);

export const IconEye = (p: IconProps) => (
  <Icon {...p}>
    <path d="M2 12s3.75-6.5 10-6.5S22 12 22 12s-3.75 6.5-10 6.5S2 12 2 12Z" />
    <circle cx="12" cy="12" r="2.75" />
  </Icon>
);

export const IconEyeOff = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 4l16 16" />
    <path d="M9.5 5.9A9.6 9.6 0 0 1 12 5.5c6.25 0 10 6.5 10 6.5a17 17 0 0 1-3.3 4.1" />
    <path d="M6.4 7.9A17 17 0 0 0 2 12s3.75 6.5 10 6.5a9.9 9.9 0 0 0 3.4-.6" />
  </Icon>
);

export const IconArrowRight = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 12h15" />
    <path d="M13 6l6 6-6 6" />
  </Icon>
);

export const IconCheck = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 12.5 9.5 18 20 6.5" />
  </Icon>
);

/** Live/ready indicator. Paired with a colour, never used alone to convey state. */
export const IconSpark = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4" />
    <path d="M6.5 6.5 9 9M15 15l2.5 2.5M17.5 6.5 15 9M9 15l-2.5 2.5" />
  </Icon>
);

/**
 * Translator card mark: an open hand between two scan brackets. Custom to
 * HandSon so the home card reads as "camera + hand + sign", not generic AI.
 */
export const IconHandScan = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 8V4.5A1.5 1.5 0 0 1 4.5 3H8" />
    <path d="M16 3h3.5A1.5 1.5 0 0 1 21 4.5V8" />
    <path d="M21 16v3.5a1.5 1.5 0 0 1-1.5 1.5H16" />
    <path d="M8 21H4.5A1.5 1.5 0 0 1 3 19.5V16" />
    <path d="M9 12V6.5a1.5 1.5 0 0 1 3 0V11" />
    <path d="M12 11V5.5a1.5 1.5 0 0 1 3 0V11" />
    <path d="M15 11.5V8a1.5 1.5 0 0 1 3 0v6.5a6.5 6.5 0 0 1-6.5 6.5H10a5 5 0 0 1-4.2-2.3L4 16" />
  </Icon>
);

/**
 * Avatar card mark: an isometric cube with a hand glyph, to say
 * "3D + sign" in one mark.
 */
export const IconCubeHand = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 2.5 21 7.25v9.5L12 21.5 3 16.75v-9.5z" />
    <path d="M3 7.25 12 12l9-4.75" />
    <path d="M12 12v9.5" />
  </Icon>
);