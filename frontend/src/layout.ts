import { createEffect, createSignal, on } from "solid-js";
import { asFlag, fromFlag, persisted } from "./state";

/** The places prose hides behind a (?). One entry per *spot on screen*, not per
 * button: the caption panel and the stage form are visible at the same time, so
 * one (?) may never speak for both. The Settings panes split the same way —
 * each `<h4>` block is its own area, since a dialog shows several at once.
 */
export const HELP_AREAS = [
  "stage",
  "caption",
  "roots",
  "defaults",
  "preprocess",
  "token",
  "models",
] as const;
export type HelpArea = (typeof HELP_AREAS)[number];

const isArea = (v: string): v is HelpArea => (HELP_AREAS as readonly string[]).includes(v);
/** Stored as the comma-joined list of what is open. An older `"1"`/`"0"` under
    the same key parses to nothing open, which is the off state it meant. */
const readAreas = (raw: string) => raw.split(",").filter(isArea);

/** Which panes are on screen and how tall the dock is — preferences that
 * survive a reload but mean nothing to the server.
 */
export function createLayout() {
  const [sidebar, setSidebar] = persisted("sidebar", true, asFlag, fromFlag);
  const [dockOpen, setDockOpen] = persisted("dock", true, asFlag, fromFlag);
  // Not `persisted`: this moves on every pointermove of a drag, so it is saved
  // once on pointerup (see `grip`) rather than at frame rate.
  const [dockH, setDockH] = createSignal(Number(localStorage.getItem("dockh")) || 320);
  /** The job log window. Not `persisted`, unlike everything else here: it is a
      look at the run happening now, not a preference about the layout, and a
      reload has no job of its own to open it on. */
  const [logOpen, setLogOpen] = createSignal(false);
  /** Which explanations are open, off by default. A (?) speaks for its own
      area only — pressing the stage bar's does not light up the caption panel
      or the Settings blurbs — and the ☰ menu's row is the one that opens every
      area at once. */
  const [helpAreas, setHelpAreas] = persisted<HelpArea[]>("help", [], readAreas, (v) =>
    v.join(","),
  );

  // A class on <body>, not a <Show>: the tree keeps its expanded folders and
  // its scroll position while it is folded away.
  createEffect(on(sidebar, (v) => document.body.classList.toggle("nosidebar", !v)));

  function grip(e: PointerEvent) {
    e.preventDefault();
    const y0 = e.clientY;
    const h0 = dockH();
    const move = (ev: PointerEvent) =>
      setDockH(Math.max(120, Math.min(window.innerHeight - 220, h0 + (y0 - ev.clientY))));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      localStorage.setItem("dockh", String(dockH()));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  return {
    sidebar,
    setSidebar,
    dockOpen,
    setDockOpen,
    toggleDock: () => setDockOpen(!dockOpen()),
    dockH,
    grip,
    logOpen,
    setLogOpen,
    helpOpen: (a: HelpArea) => helpAreas().includes(a),
    toggleHelp: (a: HelpArea) =>
      setHelpAreas((v) => (v.includes(a) ? v.filter((x) => x !== a) : [...v, a])),
    /** The menu row: on only while nothing is left to open. */
    allHelp: () => helpAreas().length === HELP_AREAS.length,
    toggleAllHelp: () =>
      setHelpAreas((v) => (v.length === HELP_AREAS.length ? [] : [...HELP_AREAS])),
  };
}

export type Layout = ReturnType<typeof createLayout>;
