import { Show, type JSX } from "solid-js";
import type { ZoomPan } from "../zoomPan";

/** The box a zoomable picture sits in: Ctrl/⌘+scroll zooms about the pointer,
    a drag pans once zoomed, a double-click gives the whole picture back. The
    handlers go on this frame and `zp.style()` on whatever is inside it, which is
    the caller's — an image, a mask overlay, the OCR boxes, the compare stage.
    `class` names the frame's own look (`frame` in the preview, `pairframe` in
    compare view); `zoomed` is added while it is zoomed in. */
export function ZoomFrame(props: {
  zp: ZoomPan;
  class?: string;
  title?: string;
  ref?: HTMLDivElement | ((el: HTMLDivElement) => void);
  children: JSX.Element;
}) {
  return (
    <div
      ref={props.ref}
      class={props.class ?? "frame"}
      classList={{ zoomed: props.zp.zoom() > 1 }}
      onWheel={props.zp.wheel}
      onPointerDown={props.zp.drag}
      onDblClick={props.zp.reset}
      title={props.title}
    >
      {props.children}
    </div>
  );
}

/** ` · 2.5×` after a frame's read-out while it is zoomed in, nothing at 1×. */
export function ZoomLevel(props: { zp: ZoomPan }) {
  return <Show when={props.zp.zoom() > 1}> · {props.zp.zoom().toFixed(1)}×</Show>;
}
