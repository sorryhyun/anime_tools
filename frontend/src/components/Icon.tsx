/** The page's line icons, drawn inline rather than as text glyphs: a glyph's
    weight, size and baseline change with whichever font the platform falls
    back to for it, so a ↻ beside a ‹ never lined up with the buttons around
    it. Every path is on a 24-unit grid and inherits the text colour, so a
    button's hover and disabled states tint it the way they tint a label. */
const PATHS = {
  refresh: ["M20 11a8 8 0 1 0-2.3 5.7", "M20 4v7h-7"],
  chevronLeft: ["M15 5l-7 7 7 7"],
  chevronRight: ["M9 5l7 7-7 7"],
} as const;

export type IconName = keyof typeof PATHS;

export function Icon(props: { name: IconName }) {
  return (
    <svg class="ico" viewBox="0 0 24 24" aria-hidden="true">
      {PATHS[props.name].map((d) => (
        <path d={d} />
      ))}
    </svg>
  );
}
