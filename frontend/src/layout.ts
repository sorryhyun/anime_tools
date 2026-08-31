import { createEffect, createSignal, on } from "solid-js";
import { asFlag, fromFlag, persisted } from "./state";

/** Which panes are on screen and how tall the dock is — the preferences that
 * survive a reload but mean nothing to the server.
 *
 * The dataset sidebar, the stage dock and the caption panel all want the same
 * space and which one wins changes by the minute, so all three are foldable and
 * the boundary between the tree and the dock is a drag.
 */
export function createLayout() {
  const [sidebar, setSidebar] = persisted("sidebar", true, asFlag, fromFlag);
  const [dockOpen, setDockOpen] = persisted("dock", true, asFlag, fromFlag);
  // Not `persisted`: this moves on every pointermove of a drag, so it is saved
  // once on pointerup (see `grip`) rather than at frame rate.
  const [dockH, setDockH] = createSignal(Number(localStorage.getItem("dockh")) || 320);
  /** One global "show the prose" preference, off by default: the stage doc and
      the Settings blurbs are behind the (?) buttons until it is on. */
  const [help, setHelp] = persisted("help", false, asFlag, fromFlag);

  // A class on <body>, not a <Show>: the tree keeps its expanded folders and
  // its scroll position while it is folded away.
  createEffect(on(sidebar, (v) => document.body.classList.toggle("nosidebar", !v)));

  /** Drag the dock's top edge. The dataset view and the stage form both want
      the vertical space, and which one wins changes by the minute. */
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
    toggleSidebar: () => setSidebar(!sidebar()),
    dockOpen,
    setDockOpen,
    toggleDock: () => setDockOpen(!dockOpen()),
    dockH,
    grip,
    help,
    toggleHelp: () => setHelp(!help()),
  };
}

export type Layout = ReturnType<typeof createLayout>;
