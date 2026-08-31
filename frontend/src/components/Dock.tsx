import { For, Show, type JSX } from "solid-js";
import type { Stage } from "../types";

/** The bottom dock: one button per panel, a drag handle for its height, and the
    open panel's body (the stage runner) as its children.

    The button strip is the stage picker, and only that: the dock is folded by
    the ▾ in the strip's corner, so pressing the open panel's own button leaves
    it where it is. Which stage inside a panel runs is picked in the body, so
    this only ever knows panels — and the body is rendered only while the dock is
    open, so a folded dock costs nothing. */
export function Dock(props: {
  open: boolean;
  height: number;
  /** The panels, in registry order, with the stages under each. */
  panels: [string, Stage[]][];
  curPanel: string;
  busy: boolean;
  onPick: (panel: string, stages: Stage[]) => void;
  onToggle: () => void;
  /** Pointer-down on the top edge; the drag itself lives in `layout.ts`. */
  onResize: (e: PointerEvent) => void;
  children?: JSX.Element;
}) {
  return (
    <div
      classList={{ dock: true, closed: !props.open }}
      style={{ "--dock-h": `${props.height}px` }}
    >
      <Show when={props.open}>
        <div class="dockgrip" onPointerDown={props.onResize} title="Drag to resize" />
      </Show>
      <div class="tabs stagetabs">
        <For each={props.panels}>
          {([p, ss]) => (
            <a
              classList={{
                sel: props.open && props.curPanel === p,
                na: !ss.some((s) => s.available),
              }}
              title={ss.map((s) => s.title).join(" · ")}
              onClick={() => props.onPick(p, ss)}
            >
              {p}
            </a>
          )}
        </For>
        <span class="sp" />
        <Show when={props.busy}>
          <span class="badge running">running</span>
        </Show>
        <button class="link" onClick={props.onToggle}>
          {props.open ? "▾" : "▴"}
        </button>
      </div>

      <Show when={props.open}>
        <div class="dockbody">{props.children}</div>
      </Show>
    </div>
  );
}
