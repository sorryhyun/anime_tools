import { createEffect, createSignal, on } from "solid-js";
import { asFlag, fromFlag, persisted } from "./state";

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
  /** One global "show the prose" preference, off by default: the stage doc and
      the Settings blurbs sit behind the (?) buttons until it is on. */
  const [help, setHelp] = persisted("help", false, asFlag, fromFlag);

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
    help,
    toggleHelp: () => setHelp(!help()),
  };
}

export type Layout = ReturnType<typeof createLayout>;
