import { For, Show, type JSX } from "solid-js";
import { t } from "../i18n";
import type { Stage } from "../types";

/** The bottom dock: one button per panel, a drag handle for its height, and the
    open panel's body (the stage runner) as its children. The strip only picks a
    panel — the dock folds by the ▾ in its corner, and which stage inside a panel
    runs is picked in the body. The body renders only while the dock is open. */
export function Dock(props: {
  open: boolean;
  height: number;
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
        <div class="dockgrip" onPointerDown={props.onResize} title={t().common.dragToResize} />
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
          <span class="badge running">{t().common.running}</span>
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
